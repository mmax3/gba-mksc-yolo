from BHServer import BHServer

# Start the TCP server
server = BHServer(
    # Server Settings
    ip = "127.0.0.1",
    port = 1337,
    # Data Settings
    use_grayscale = False,  # Store screenshots in grayscale
    system = "GBA",  # Initialize server.controls to standard N64 controls
    # Client Settings
    mode = "HUMAN",
    update_interval = 5,  # Update to server every 5 frames
    frameskip = 1,
    speed = 100,  # Emulate at 6399% original game speed (max)
    sound = False,  # Turn off sound
    rom = "ROMs/Mario Kart - Super Circuit.gba",  # Add a game ROM file
    saves = {"GBA/State/Mario Kart - Super Circuit (Europe).mGBA.QuickSave1.State": 1}  # Add a save state
)
server.start()


def update(self):
    #print(self.client_started())
    if self.client_started():
        print(self.actions)
        print(self.screenshots[self.actions - 1].shape)
    if self.controls["B"]:        # If B button is pressed prints all input states
        print(self.controls)
    
    actions = self.actions              # Grab number of times update() has been called
    ss = self.screenshots[actions - 1]  # Grab the latest screenshot (numpy.ndarray)

    #self.controls["A"] = True    # Press the A button on Player 1's controller, mode has to be other than "HUMAN"
    x_type = self.data["x"][0]    # Get type of variable x: "INT". Set by client
    x = self.data["x"][1]         # Get value of variable x: 512. Set by client
    """
    if actions == 20:
        self.save_screenshots(0, actions - 1, "my_screenshot")
    elif actions == 40:
        self.new_episode()      # Reset the emulator, actions = 0, ++episodes
        if self.episodes == 3:  # Stop client after 3 episodes
            self.exit_client()
    """

# Replace the server's update function with ours
BHServer.update = update
print(f"Server ready at IP:{server.ip} port:{server.port}")
print(f"Run EmuHawk.exe with these parameters:")
print(f"--socket_ip={server.ip} --socket_port={server.port} --url_get=http://{server.ip}:9876/get --url_post=http://{server.ip}:9876/post")
# Optional loop that can be implemented. Runs in main thread rather than server
# while True:
#     pass
