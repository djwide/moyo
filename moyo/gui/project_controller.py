"""Qt helper that tracks the GUI's current moyo project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from moyo.project import (
    MoyoProject,
    create_project,
    list_projects,
    load_saved_project,
    save_current_project,
)


class ProjectController(QObject):
    """Holds the selected project and notifies tabs when it changes."""

    changed = pyqtSignal(object)  # MoyoProject | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current: Optional[MoyoProject] = None
        self._extra: list[Path] = []
        saved = load_saved_project()
        if saved is not None:
            self._current = saved
            if saved.root.parent != saved.root:
                self._extra.append(saved.root)

    @property
    def current(self) -> Optional[MoyoProject]:
        return self._current

    def projects(self) -> list[MoyoProject]:
        extra = list(self._extra)
        if self._current is not None:
            extra.append(self._current.root)
        return list_projects(extra=extra)

    def set_project(self, project: Optional[MoyoProject], *, persist: bool = True) -> None:
        self._current = project
        if project is not None:
            self._extra.append(project.root)
        if persist:
            save_current_project(project)
        self.changed.emit(project)

    def select_name(self, name: str) -> Optional[MoyoProject]:
        for proj in self.projects():
            if proj.name == name or str(proj.root) == name:
                self.set_project(proj)
                return proj
        return None

    def open_folder(self, path: Path) -> MoyoProject:
        project = MoyoProject.from_path(path).ensure()
        self._extra.append(project.root)
        self.set_project(project)
        return project

    def create(self, name: str) -> MoyoProject:
        project = create_project(name)
        self.set_project(project)
        return project
