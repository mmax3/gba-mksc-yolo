--Bizhawk: Common Lua Functions
--Purpose is to show functional examples of common Lua functions within Bizhawk
--Note that all functions can be viewed at http://tasvideos.org/Bizhawk/LuaFunctions.html
--Also note that in the Lua Console window -> Help -> Lua Functions window you get a list of functions with syntax and examples.
--Author: CoolHandMike


while true do

	memory.usememorydomain("WRAM") --Specify Memory Domain

		--COMMON DRAW FUNCTIONS	
	gui.drawLine( 180, 0, 180, 230, "Black" )
	gui.drawRectangle( 16, 170, 200, 45, 0x007F00FF, 0x7F7F7FFF )
	gui.drawText( 16 , 90, "Some message" , "Red")
	gui.text(10,185, "Text with default formatting") -- Better performance over gui.drawText()
	gui.drawBox( 16, 0, 250, 150, "Black", 0xE67F7FFF )
	gui.drawImage( "C:\pet3.png", 0, 110, 18, 24, false )
	gui.drawImage( "feather.png", 64, 110, 18, 24, false )
	

	--COMMON READ FUNCTIONS

	local val =  mainmemory.readbyte(0x0359)  --reads unsigned byte (8 bits)
	local val_u8 =  mainmemory.read_u8(0x0360) --unsigned 8 bits
	local val_s8 =  mainmemory.read_s8(0x0361)  --signed 8 bits

	local val_u16le =  mainmemory.read_u16_le(0x0362) --read unsigned 16 bits little endian
	local val_s16le =  mainmemory.read_s16_le(0x0364)  --reads signed 16 bits
	local val_u32le =  mainmemory.read_u32_le(0x0370)  --reads unsigned 32 bits
	local val_s32le =  mainmemory.read_s32_le(0x0366)  --reads signed 32 bits

	local val_u16be =  mainmemory.read_u16_be(0x0362) --read unsigned 16 bits big endian
	local val_s16be =  mainmemory.read_s16_be(0x0364)  --reads signed 16 bits
	local val_u32be =  mainmemory.read_u32_be(0x0370)  --reads unsigned 32 bits
	local val_s32be =  mainmemory.read_s32_be(0x0366)  --reads signed 32 bits

	local nlmairea = mainmemory.readbyterange( 0x100, 64 ) --best performance, but hardest to deal with. may want to use brunovalads' byterange_decoder http://tasvideos.org/userfiles/info/67926325655324243


	-- WRITE FUNCTIONS
	
	--Replace above text "read" with "write", and add a parameter to write. and you will get all the options of writing bytes.
	mainmemory.writebyte(0x0362, 28)
	mainmemory.write_u8(0x0360, 31) --unsigned 8 bits
	mainmemory.write_s8(0x0361, 43)  --signed 8 bits

	mainmemory.write_u16_be(0x0362, 500) --read unsigned 16 bits big endian
	mainmemory.write_s16_be(0x0364, 500)  --reads signed 16 bits
	mainmemory.write_u32_be(0x0370, 20000)  --reads unsigned 32 bits
	--ETC
	--//mainmemory.writebyterange(nluatable memoryblock)


	--COMMON BIT FUNCTIONS
	--//bit.band(uint  val, uint  amt) --bitwise and. 
	 local uibitban = bit.band( 1, 4 );   
	 gui.drawText( 16 , 0, "bit.band( 1, 4 ) returns " .. bit.band( 1, 4 ) , "Red");

	 local uibitrsh = bit.rshift( 8, 1 );
	 gui.drawText( 16 , 16, "bit.rshift( 8, 1 ) returns " ..  bit.rshift( 8, 1 ), "Red");

	
	 if bit.check( 2, 1 )  then
		 gui.drawText( 16 , 32, "bit.check( 2, 1 ) returns " .."true" , "Red");
	 else
		 gui.drawText( 16 , 32, "bit.check( 2, 1 ) returns " .."false" , "Red");
	 end;
	
	-- -- --//bit.clear(uint  num, int pos)-- position 0 is the rightmost bit.
	local lobitcle = bit.clear( 3, 0); 
	gui.drawText( 16 , 48, "bit.clear( 3, 0) returns " .. lobitcle , "Red");
	local uibitset = bit.set( 4, 0 ); 
	gui.drawText( 16 , 60, "bit.set( 4, 0 ) returns " .. uibitset , "Red");
	local uibitbxo = bit.bxor( 1, 1 );
	gui.drawText( 16 , 76, "bit.bxor( 1, 1 ) returns " .. uibitbxo , "Red");
	--ETC
	
	--SCREEN FUNCTIONS
	   client.screenwidth() 
	   client.screenheight()

	 --//client.SetGameExtraPadding(int left, int top, int right, int bottom)
	client.SetGameExtraPadding( 0, 10, 230, 20 );

	local userinput = input.getmouse()
	gui.drawText( 260 , 110,  "mouse x : " .. userinput.X.. " ; mouse y: " ..  userinput.Y , "Red");
	
	
	--SAVESTATE FUNCTIONS
	
	--//savestate.loadslot(int slotnum, [bool suppressosd = False])	
	if false == true then --never actually run, just have it uncommented for readability
		savestate.saveslot( 7 );
		savestate.loadslot( 8 );
	end 
	
	
	--JOYPAD FUNCTIONS
	--This KeyCheck function is helpful to determine the controller names when trying to access them in a table.  http://tasvideos.org/userfiles/edit/67926325655324243@user
	--joypad.getimmediate works similar to joypad.get()
	if joypad.get()["P1 Right"]==true then 
		local p1 = joypad.get(1)
		p1["P1 B"]=true
		joypad.set(p1)
		gui.drawText( 15, 115, "Reading input Right", "Red");
	end
	
	
	emu.frameadvance();
end