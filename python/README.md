# OpenCollidoscope (Python)

The original app is a **C++ / Cinder** program. It does not run on current Python. This package is a port of that engine so it runs on **CPython 3.11+** (Raspberry Pi OS Bookworm and later).

It keeps the same model: two stereo waves, 2-second record buffers, selection-based granular playback, grain overlap, low-pass filter, MIDI from a Teensy (or any controller), and a computer-keyboard fallback.

## Raspberry Pi

```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-pygame python3-numpy libportaudio2 libasound2-dev
cd python
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install sounddevice python-rtmidi
python3 -m opencollidoscope 1024 768
```

PortAudio does not follow the desktop default sink. A combo keyboard or screen that exposes its own mic can become the automatic recorder. The commands to list devices and pin them are at the end of this file.

`--system-site-packages` lets the venv use the OS `pygame` / `numpy` builds, which is the reliable path on Raspberry Pi (pip wheels for pygame often fail on ARM).

If `apt` has no `python3-pygame`, install pygame into the venv instead:

```bash
python3 -m pip install pygame sounddevice python-rtmidi numpy
```

## Controls

| Input | Action |
| --- | --- |
| `Tab` | Active wave (1 or 2) |
| `R` / `T` | Record wave 1 / wave 2 from the mic |
| `Space` | Loop the selection on the active wave |
| `A` `D` or arrows | Selection start |
| `W` `S` | Selection length |
| `9` `0` | Grain duration (overlap) |
| `Z`…`M` | Notes from C3 (same pitch map as MIDI 48+) |
| Drag | Selection start + filter (vertical) |
| Scroll | Selection length |
| `F1` | Input debug overlay |
| MIDI note on/off | Voices (6 per wave), middle C = recorded pitch |
| Pitch bend | Selection start. 0–127 spans the whole wave (127 is the last chunk) |
| CC 1 / 2 / 4 / 5 / 7 | Size, grain duration, loop, record, filter |

## Debug

Start with the overlay on, and leave an SSH session running so you can still read the log if the keyboard drops:

```bash
python3 -m opencollidoscope --debug 1024 768
tail -f opencollidoscope-debug.log
```

`F1` toggles the overlay. `--debug-log /tmp/opencollidoscope-debug.log` chooses the file.

The panel is there to separate three different failures:

- **Kernel keyboards** goes to `NONE` when Linux has actually dropped the USB keyboard. Focus can still say `yes` for a moment, but new key events stop.
- **key-focus NO** means the device is still there and pygame is no longer receiving keys (the window lost keyboard focus).
- **ch1 / ch2 hits** count every MIDI message. `len dur loop rec filt` are CC 1, 2, 4, 5, and 7. A control that stays at 0 is not arriving. A control whose count climbs while the matching `wave` line does not change was ignored. Pitch bend is scaled from MIDI 0–127 across the wave; the recent line shows `level N/127`. Any other controller number is listed under `other`.

`wave1` / `wave2` are the values the engine is using right now: selection start, selection size, grain duration, filter, loop, and record. Filter and pitch-bend messages are counted in full; the recent log keeps the latest change a few times a second so a fader does not flood the file.

Hardware MIDI is optional. Without `python-rtmidi` or a controller, the computer keyboard and mouse still play the instrument.

## What this is not

It does not compile the Cinder app, flash Teensy firmware, or replace JACK with Cinder’s audio graph. JACK can still sit in front of PortAudio via `pipewire-jack` if you want that routing on current Raspberry Pi OS.

## Audio config

List the devices, then save the input and output you want. Use a unique part of the device name from that list. An index also works, but it changes when another USB device is plugged in. `usb` is too short once two USB audio devices are connected.

```bash
python3 -m opencollidoscope --list-audio
python3 -m opencollidoscope --input Scarlett --output Scarlett --save-audio
```

`--save-audio` writes `~/.config/opencollidoscope/audio.conf`. Later launches, including a desktop shortcut with no extra arguments, use that file. `--input` and `--output` on the command line override the file for that one run. Press `F1` to see the open devices and whether each one came from `command`, `config`, or `auto`.
