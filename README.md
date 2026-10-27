This is a fix of the original OpenCollidoscope.   It skips the cinder compile, and uses current python.   Took a day working with Cursor AI to convert to python and fix the conversion.  

I saw the original project on a Dr Mix video where the original creators demonstrated it.  http://collidoscope.io/  https://www.youtube.com/watch?v=9XMfKYVu_fg

My setup is a RPi4 running latest Raspbian (as of 25 Oct 2025) and Python3.  My soundcard is a Focusrite Scarlett 2i4.  
The RPi seems to want to default to HDMI audio output (input was automatically from the Scarlett) so I had to include "--output usb" as part of the command.  If you are using HDMI audio then you can skip that.

I am not using the sliders etc for controls...just midi...the teensy midi input is untested, but since it is simply sending midi over USB, it shouldn't have a problem.  You will have to dig around the original repository to get the docs etc...if they are missing I can add them here.

The Dr Mix video shows the 2 wave version, I only wanted the 1 wave version. (note the instructions below on changing that)  I am running my display at 1024x768, so this program runs at fullscreen.  More pixels seems to add more load, but you can try running at 1920x1080 on your display and in the program.

---
Instructions from Cursor AI follows.  BTW modern python almost forces you to use a Virtual Environment (venv) so I followed that convention.  (for the record, I did not install this way (my fork did not yet exist)..so can't tell you if all the instructions will work in your case)

Install it from your fork, on the python-port branch. On the Pi, replace /home/pi if your home directory is different. 


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
--system-site-packages uses the Raspberry Pi OS builds of pygame and numpy. Set NUM_WAVES in opencollidoscope/config.py before you start: 1 uses the first input only, 2 sends input 1 to wave 1 and input 2 to wave 2. Leave the desktop volume output on HDMI so the Scarlett stays with this program.

To put an icon on the desktop, save this as /home/pi/Desktop/opencollidoscope.desktop:


[Desktop Entry]
Type=Application
Name=OpenCollidoscope
Comment=Granular sampler
Exec=/home/pi/opencollidoscope/python/.venv/bin/python -m opencollidoscope --output usb 1024 768
Path=/home/pi/opencollidoscope/python
Terminal=false
Categories=AudioVideo;Audio;
Then mark it executable:


chmod +x /home/pi/Desktop/opencollidoscope.desktop
If the desktop asks you to trust the launcher, confirm that. Copying the same file to /home/pi/.local/share/applications/ also puts it in the application menu.
