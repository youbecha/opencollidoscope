
<img width="3840" height="2160" alt="image0" src="https://github.com/user-attachments/assets/b0575e5a-1308-4798-9a68-b0ed466bf5f4" />

(note: debug window in screenshot is off by default)

This is a fix of the original OpenCollidoscope.   It skips the cinder compile, and uses current python.   Took a day working with Cursor AI to convert to python and fix the conversion.  

I saw the original project on a Dr Mix video where the original creators demonstrated it.  http://collidoscope.io/  https://www.youtube.com/watch?v=9XMfKYVu_fg

My setup is a RPi4 running latest Raspbian (as of 25 Oct 2025) and Python3.  My soundcard is a Focusrite Scarlett 2i4.  
The RPi seems to want to default to HDMI audio output (input was automatically from the Scarlett) so I had to include "--output usb" as part of the command. 
If you are using HDMI audio then you can skip that.

I am not using the sliders etc for controls...just midi...the teensy midi input is untested, but since it is simply sending midi over USB, it shouldn't have a problem.  You will have to dig around the original repository to get the docs etc...if they are missing I can add them here.

Here is the Midi info.  Midi channel 1 is for wave1 or single wave. Channel 2 is for wave2 in a two waveform setup.  So in a two waveform setup Midi channel 2 also needs controls with the following CC#s.  I have 4 rotary knobs and the transport controls on my old Alesis QX49...but you could even make your own teensy midi controller if you didn't want to build the complex slider system of the original.

For each Midi channel you need the following CCs (I am only using 1 midi channel)\
  CC 145 = position (Pitchbend control)\
  CC 1 = length (modulation wheel)\
  CC 2 = duration\
  CC 7 = filter\
  CC 4 = play (just plays the waveform as is...your keyboard will pitch the waveform and play it)\
  CC 5 = record (this is how you command the software to do an audio capture off your selected input device)\


The Dr Mix video shows the 2 wave version, I only wanted the 1 wave version. (note the instructions below on changing that)  I am running my display at 1024x768, so this program runs at fullscreen.  More pixels seems to add more load, but you can try running at 1920x1080 on your display and in the program.

You don't have to use the Starlett...I had one, so I used it.  For my 2 channel input, if using single wave setup, then channel one is the input, if a dual wave setup then channel 2 goes to wave 2.  (can you imagine the drama, when the original fix put the single wave input as channel 2 and I am trying to figure out why no recorded audio...?)

IF you have a single channel input, it should go to both waves in a dual wave setup.

---
Instructions from Cursor AI follows.  BTW modern python almost forces you to use a Virtual Environment (venv) so I followed that convention.  (for the record, I did not install this way (my fork did not yet exist)..so can't tell you if all the instructions will work in your case)

Install it from your fork, on the python-port branch. On the Pi, replace /home/pi if your home directory is different. 

```
sudo apt update
sudo apt install python3-pip python3-venv python3-pygame python3-numpy libportaudio2 libasound2-dev git
cd /home/pi
git clone --branch python-port https://github.com/youbecha/opencollidoscope.git
cd opencollidoscope/python
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install sounddevice python-rtmidi
python3 -m opencollidoscope --list-audio
python3 -m opencollidoscope --output usb 1024 768
```
--system-site-packages uses the Raspberry Pi OS builds of pygame and numpy. Set NUM_WAVES in opencollidoscope/config.py before you start: 1 uses the first input only, 2 sends input 1 to wave 1 and input 2 to wave 2. Leave the desktop volume output on HDMI so the Scarlett stays with this program.

To put an icon on the desktop, save this as /home/pi/Desktop/opencollidoscope.desktop:

```
[Desktop Entry]
Type=Application
Name=OpenCollidoscope
Comment=Granular sampler
Exec=/home/pi/opencollidoscope/python/.venv/bin/python -m opencollidoscope --output usb 1024 768
Path=/home/pi/opencollidoscope/python
Terminal=false
Categories=AudioVideo;Audio;
```
Then mark it executable:
```
chmod +x /home/pi/Desktop/opencollidoscope.desktop
```
If the desktop asks you to trust the launcher, confirm that. Copying the same file to /home/pi/.local/share/applications/ also puts it in the application menu.




----------------------

PS, the way the original Collidoscope plays is the keyboard changes the rate AND pitch...

-if you do not want the pitch to change only the rate with the keyboard...then note the following message from Cursor AI and edit accordingly.

-----------------------
The edit is in WaveEngine.note_on in engine.py. That is the only place a key becomes a playback rate.


engine.py
Ln 139–149
```
def note_on(self, midi_note: int) -> None:
    rate = midi_note_ratio(midi_note)
    for i, assigned in enumerate(self.midi_notes):
        if assigned == midi_note:
            self.voices[i].note_on(rate)
            return
    for i, assigned in enumerate(self.midi_notes):
        if assigned == NO_MIDI_NOTE:
            self.voices[i].note_on(rate)
            self.midi_notes[i] = midi_note
            return
```

rate is passed into PGranular.note_on, which stores it as the pitch. 1.0 is the recorded pitch. Middle C already produces 1.0.


dsp.py
Ln 260–265
```
def note_on(self, rate: float) -> None:
    if self.env.state is EnvState.IDLE:
        if self.trigger_rate < MIN_GRAIN_DURATION:
            self.trigger_rate = MIN_GRAIN_DURATION
        self.set_grains_rate(rate)
        self.env.state = EnvState.ATTACK
```
Speed is the other number, trigger_rate. It is set from the selection length here, and a key never touches it:


dsp.py
Ln 253–258
```
def set_selection_size(self, size: int) -> None:
    size = max(size, MIN_GRAIN_DURATION)
    self.trigger_rate = size
    self.grains_duration = max(
        MIN_GRAIN_DURATION, int(round(size * self.grains_duration_coeff))
    )
```
For speed-only keys, note_on in engine.py would pass 1.0 into voices[i].note_on and multiply that voice's trigger_rate by rate before the note starts. 

Leaving note_on(rate) as it is now keeps the original pitch-and-speed behavior. 
```
self.voices[i].note_on(rate)
```
The loop calls note_on(1.0) just above this function, so it stays at the recorded pitch either way. (so you could put the rate at any preset amount for effect)
```
self.voices[i].note_on(1.0)
```

------------------------

## Audio config

A combo keyboard or screen can expose its own microphone (for example KTMicro KT USB Audio). The program records from that device unless the input and output are pinned. List the devices, then save the ones you want. Use a unique part of the name from the list. `usb` matches every USB audio device once more than one is plugged in.

```bash
cd /home/pi/opencollidoscope/python
source .venv/bin/activate
python3 -m opencollidoscope --list-audio
python3 -m opencollidoscope --input Scarlett --output Scarlett --save-audio
```

`--save-audio` writes `~/.config/opencollidoscope/audio.conf`. Later launches, including the desktop icon, use that file, so `--output usb` is no longer needed on the command or in the desktop file. `--input` or `--output` on the command line still overrides the file for that one run. Press `F1` to see which devices are open and whether each one came from `command`, `config`, or `auto`.
