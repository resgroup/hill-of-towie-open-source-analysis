import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import find_dotenv

from hot_open import settings


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Neutralise every input _load_env reads, and point REPO_ROOT at a throwaway dir.

    Without the REPO_ROOT patch these tests would pick up the real repo's own .env, whose
    presence differs per machine.
    """
    for var in ("HOT_OPEN_DATA_DIR", "HOT_OPEN_CACHE_DIR", "HOT_OPEN_OUTPUT_DIR", "HOT_OPEN_DOTENV"):
        monkeypatch.delenv(var, raising=False)
    repo_root = tmp_path / "lib_repo"
    repo_root.mkdir()
    monkeypatch.setattr(settings, "REPO_ROOT", repo_root)
    monkeypatch.setattr(settings, "REPO_NAME", repo_root.stem)
    settings._load_env.cache_clear()  # noqa: SLF001
    return tmp_path


def _write_dotenv(directory: Path, **values: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()))


class TestDotenvPrecedence:
    """.env resolution must follow the application, not this library's own location.

    ``load_dotenv()`` resolves through ``find_dotenv()``, which walks up from the *calling*
    file -- settings.py. So an application that installs hot_open and writes its own .env was
    ignored with no warning: an editable install silently loaded the hot_open checkout's .env
    instead, and a wheel install found none at all, having walked site-packages to the root.
    """

    def test_application_dotenv_in_cwd_is_used(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        monkeypatch.chdir(app)
        assert settings.get_data_dir() == app / "from_app"

    def test_application_dotenv_found_from_a_subdirectory(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        nested = app / "deep" / "deeper"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        assert settings.get_data_dir() == app / "from_app"

    def test_application_dotenv_beats_this_repos_own(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The regression this fix exists for: previously the library's own .env won.
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        _write_dotenv(settings.REPO_ROOT, HOT_OPEN_DATA_DIR=str(isolated_env / "from_lib"))
        monkeypatch.chdir(app)
        assert settings.get_data_dir() == app / "from_app"

    def test_this_repos_own_dotenv_is_the_fallback(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Scripts run inside this repo must keep working exactly as before.
        unrelated = isolated_env / "unrelated"
        unrelated.mkdir()
        _write_dotenv(settings.REPO_ROOT, HOT_OPEN_DATA_DIR=str(isolated_env / "from_lib"))
        monkeypatch.chdir(unrelated)
        if find_dotenv(usecwd=True):
            pytest.skip("a .env exists above the temp dir, so the cwd branch cannot be ruled out here")
        assert settings.get_data_dir() == isolated_env / "from_lib"

    def test_explicit_dotenv_path_wins_over_both(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        explicit = isolated_env / "explicit.env"
        explicit.write_text(f"HOT_OPEN_DATA_DIR={isolated_env / 'from_explicit'}\n")
        monkeypatch.setenv("HOT_OPEN_DOTENV", str(explicit))
        monkeypatch.chdir(app)
        settings._load_env.cache_clear()  # noqa: SLF001
        assert settings.get_data_dir() == isolated_env / "from_explicit"

    def test_real_environment_variable_wins_over_every_file(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        monkeypatch.setenv("HOT_OPEN_DATA_DIR", str(isolated_env / "from_real_env"))
        monkeypatch.chdir(app)
        settings._load_env.cache_clear()  # noqa: SLF001
        assert settings.get_data_dir() == isolated_env / "from_real_env"

    def test_a_later_file_only_fills_what_an_earlier_one_omits(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # override=False throughout, so the files layer rather than replace each other.
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        _write_dotenv(
            settings.REPO_ROOT,
            HOT_OPEN_DATA_DIR=str(isolated_env / "from_lib"),
            HOT_OPEN_CACHE_DIR=str(isolated_env / "cache_from_lib"),
        )
        monkeypatch.chdir(app)
        assert settings.get_data_dir() == app / "from_app"
        assert settings.get_cache_dir() == isolated_env / "cache_from_lib"


class TestLoadEnvIsCached:
    def test_repeated_getter_calls_load_env_once(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        app = isolated_env / "app_repo"
        _write_dotenv(app, HOT_OPEN_DATA_DIR=str(app / "from_app"))
        monkeypatch.chdir(app)
        settings.get_data_dir()
        settings.get_cache_dir()
        settings.get_data_dir()
        info = settings._load_env.cache_info()  # noqa: SLF001
        assert info.misses == 1
        assert info.hits == 2


class TestFallbackSurvivesMissingMainFile:
    """The repo fallback must not depend on frame introspection.

    ``find_dotenv`` abandons its frame walk and uses the cwd whenever ``__main__`` has no
    ``__file__`` -- ``python -c``, a REPL, or a Jupyter notebook. A frame-based fallback would
    therefore stop working in notebooks, silently. Hence REPO_ROOT / ".env" is an explicit path.
    """

    def test_repo_dotenv_still_found_without_a_main_file(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "lib_repo"
        pkg = repo_root / "src" / "hot_open_under_test"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").touch()
        (pkg / "settings.py").write_text(Path(settings.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        (repo_root / ".env").write_text(f"HOT_OPEN_DATA_DIR={tmp_path / 'from_lib'}\n")
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()

        # -c, so __main__ has no __file__ -- exactly the notebook case.
        code = (
            f"import sys; sys.path.insert(0, r'{repo_root / 'src'}')\n"
            "from hot_open_under_test.settings import get_data_dir\n"
            "print(get_data_dir())\n"
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"HOT_OPEN_DATA_DIR", "HOT_OPEN_CACHE_DIR", "HOT_OPEN_DOTENV"}
        }
        result = subprocess.run(  # noqa: S603 -- sys.executable and a literal snippet, no external input
            [sys.executable, "-c", code], cwd=unrelated, env=env, capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == str(tmp_path / "from_lib")
