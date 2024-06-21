import math
from collections.abc import Callable, Iterable
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

import numpy
import pyaudio


class Threaded:
    def __init__(self) -> None:
        self._thread: Thread | None = None

    def set_target(self, target: Callable[[...], object | None], args: Iterable) -> None:
        self._thread = Thread(target=target, args=args)

    def start(self) -> None:
        self._thread.start()

    def stop(self):
        self._thread.join()


def tempered(note: int, base: float) -> float:
    return base * 2 ** (note / 12)


class Acetza:
    _rations = numpy.array(object=[1 / 1, 256 / 243, 9 / 8, 32 / 27, 81 / 64, 4 / 3, tempered(note=6, base=1.0), 3 / 2,
                                   128 / 81, 27 / 16, 16 / 9,
                                   243 / 128], dtype=numpy.float64)
    base: float = 16.0

    @staticmethod
    def _ration(note: int):
        return Acetza._rations[note % 12]

    @staticmethod
    def _power(note: int):
        return 2 ** note // 12

    @staticmethod
    def freq(note: int) -> float:
        return Acetza.base * Acetza._ration(note) * Acetza._power(note)


class Synth:
    def note_on(self, key: int, velocity: int) -> None:
        ...

    def note_off(self, key: int, velocity: int) -> None:
        ...


class Locked[T]:
    def __init__(self, value: T) -> None:
        self._value: T = value
        self._lock = Lock()

    def get(self) -> T:
        with self._lock:
            value: T = self._value
        return value

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value


class Constants:
    CHANNELS: int = 2
    FRAME_RATE: int = 48_000
    BLOCK_SIZE: int = 512


def frame_to_time(frame: int) -> float:
    return frame / Constants.FRAME_RATE


def time_to_frame(time: float) -> int:
    return int(time * Constants.FRAME_RATE)


def sin(part: float) -> float:
    return math.sin(2.0 * math.pi * part)


class Block:
    def __init__(self):
        self._lock = Lock()
        self.samples = numpy.zeros((Constants.CHANNELS, Constants.BLOCK_SIZE), dtype=numpy.float64)
        self.ready = True

    def lock(self) -> Lock:
        return self._lock


class Sinner:
    def __init__(self, frequency: float = 360.0) -> None:
        self._frequency: float = frequency
        self._frame = 0

    def process(self, block: Block):
        for frame in range(Constants.BLOCK_SIZE):
            time: float = frame_to_time(self._frame)
            part: float = time * self._frequency % 1.0
            sample: float = sin(part=part)
            for channel in range(Constants.CHANNELS):
                with block.lock():
                    block.samples[channel][frame] += sample
            self._frame += 1


class Buffer:
    def __init__(self, queue: Queue) -> None:
        self._blocks: list[Block] = [Block(), Block(), Block()]
        self._current: int = 0
        self._queue = queue

    def current(self) -> Block:
        return self._blocks[self._current]

    def advance(self) -> None:
        self._current = (self._current + 1) % len(self._blocks)

    def next(self, samples: numpy.ndarray) -> None:
        if not self.current().ready:
            samples.fill(0.0)
            return
        index: int = 0
        for frame in range(Constants.BLOCK_SIZE):
            for channel in range(Constants.CHANNELS):
                samples[index] = self.current().samples[channel][frame]
                index += 1
        self.current().samples.fill(0.0)
        self.current().ready = False
        self._queue.put(item=self.current())
        self.advance()


class Audio(Threaded):
    def __init__(self, playing: Locked[bool], buffer: Buffer) -> None:
        super().__init__()
        self._playing = playing
        self._buffer: Buffer = buffer
        self.set_target(target=self._loop, args=[None])

    def _loop(self, _: Any) -> None:
        audio: pyaudio.PyAudio | None = None
        stream: pyaudio.Stream | None = None
        samples = numpy.empty(Constants.CHANNELS * Constants.BLOCK_SIZE, dtype=numpy.float32)
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(output=True, rate=Constants.FRAME_RATE, channels=Constants.CHANNELS,
                                format=pyaudio.paFloat32, frames_per_buffer=Constants.BLOCK_SIZE)
            while self._playing.get():
                self._buffer.next(samples=samples)
                stream.write(frames=samples.tobytes(), num_frames=Constants.BLOCK_SIZE)
        finally:
            if stream is not None:
                stream.close()
            if audio is not None:
                audio.terminate()


class MessageHandler(Threaded):
    def __init__(self, playing: Locked[bool], queue: Queue) -> None:
        super().__init__()
        self._queue = queue
        self._playing = playing
        self.set_target(target=self._loop, args=[None])

    def _loop(self, _: Any) -> None:
        while self._playing.get():
            ...


class Automaton:
    def __init__(self, queue: Queue) -> None:
        self._queue = queue


class Processor(Threaded):
    def __init__(self, playing: Locked[bool], queue: Queue) -> None:
        super().__init__()
        self._playing = playing
        self._queue = queue
        self.set_target(target=self._loop, args=[None])

    def _loop(self, _: Any) -> None:
        synth = Sinner()
        while self._playing.get():
            try:
                block: Block = self._queue.get(timeout=1.0)
                # print("block arrived", block)
                synth.process(block=block)
                block.ready = True
            except Empty:
                ...


def main() -> None:
    process_queue = Queue()
    playing = Locked(value=False)
    buffer = Buffer(queue=process_queue)
    audio = Audio(playing=playing, buffer=buffer)
    processor = Processor(playing=playing, queue=process_queue)
    print("Press enter to exit")
    playing.set(value=True)
    processor.start()
    audio.start()
    input()
    playing.set(False)
    audio.stop()
    processor.stop()


def test() -> None:
    limit: int = 32
    for n in range(-limit, limit):
        print(n, Acetza._power(n))


if __name__ == "__main__":
    main()
