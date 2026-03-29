from __future__ import annotations

import abc

from connect4.board import Board


class BasePlayer(abc.ABC):
    @abc.abstractmethod
    def choose_move(self, board: Board) -> int:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset(self) -> None:
        pass
