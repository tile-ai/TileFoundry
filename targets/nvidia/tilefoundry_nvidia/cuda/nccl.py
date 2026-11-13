"""Bootstrap caller-owned communicators through an already linked NCCL program."""

import ctypes
from pathlib import Path


class UniqueId(ctypes.Structure):
    """NCCL's by-value bootstrap token ABI."""

    _fields_ = [("internal", ctypes.c_byte * 128)]


class Library:
    """Use the NCCL dependency selected when the supplied program was linked.

    The caller transports tokens and selects the CUDA device. This object does
    not discover installations, create process groups or alter the environment.
    """

    def __init__(self, library_path: str | Path):
        self.library = ctypes.CDLL(str(Path(library_path).resolve(strict=True)))
        try:
            self.error_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int)(
                ("ncclGetErrorString", self.library))
            self.get_unique_id = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(UniqueId))(
                ("ncclGetUniqueId", self.library))
            self.init_rank = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                            ctypes.c_int, UniqueId, ctypes.c_int)(
                ("ncclCommInitRank", self.library))
            self.destroy_comm = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(
                ("ncclCommDestroy", self.library))
        except AttributeError as error:
            raise RuntimeError(f"program has no linked NCCL bootstrap symbols: {library_path}") from error

    def check(self, status):
        if status:
            raise RuntimeError(self.error_string(status).decode())

    def unique_id(self) -> bytes:
        token = UniqueId()
        self.check(self.get_unique_id(ctypes.byref(token)))
        return bytes(token)

    def create(self, size: int, token: bytes, rank: int) -> int:
        if not 0 <= rank < size or len(token) != ctypes.sizeof(UniqueId):
            raise ValueError("NCCL bootstrap requires a valid local rank and unique ID")
        value = ctypes.c_void_p()
        self.check(self.init_rank(ctypes.byref(value), size, UniqueId.from_buffer_copy(token), rank))
        return value.value

    def destroy(self, communicator: int) -> None:
        self.check(self.destroy_comm(communicator))


__all__ = ["Library"]
