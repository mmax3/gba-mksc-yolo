function round(num, numDecimalPlaces)
	local mult = 10^(numDecimalPlaces or 0)
	return math.floor(num * mult + 0.5) / mult
end

function get_baseline(frames)
	i = frames
	client.reboot_core()
	t = os.clock()

	while i > 0 do
		emu.frameadvance()
		i = i - 1
	end
	baseline = os.clock() - t
	print("Baseline:            " .. round(baseline, 3) .. " secs")
	return baseline
end

function test_mmf(frames)
	i = frames
	client.reboot_core()
	t = os.clock()
	while i > 0 do
		emu.frameadvance()
		comm.mmfScreenshot()
		i = i - 1
	end
	print("Memory mapped files: " .. round((os.clock() - t ), 3) .. " secs")
end

function test_http(frames)
	print("Testing HTTP server")
	client.reboot_core()
	i = frames
	t = os.clock()
	
	while i > 0 do
		emu.frameadvance()
		comm.httpTestGet()
		i = i - 1
	end
	print("HTTP get:            " ..  round((os.clock() - t ), 3) .. " secs")
	
	client.reboot_core()
	i = frames
	t = os.clock()
	
	while i > 0 do
		emu.frameadvance()
		comm.httpPostScreenshot()
		i = i - 1
	end
	print("HTTP post:           " ..  round((os.clock() - t ), 3) .. " secs")

end

function test_socket(frames)

	i = frames
	client.reboot_core()
	resp = comm.socketServerScreenShotResponse()
	if resp ~= 'ack' then
		print("Socket server: did not respond correctly, excpected `ack`, got: " .. resp)
		return
	end
	t = os.clock()
	while i > 0 do
		emu.frameadvance()
		comm.socketServerScreenShot()
		i = i - 1
	end
	print("Socket server:       " .. round((os.clock() - t ), 3) .. " secs")
end

function test_socketresponse(frames)
	best_time = -100
	timeouts = {1, 2, 3, 4, 5, 10, 20, 25, 50, 100, 250, 500, 1000, 10000}
	comm.socketServerSetTimeout(10000)
	resp = comm.socketServerScreenShotResponse()
	print("Trying to find minimal timeout for Socket server")
	for _, timeout in ipairs(timeouts) do
		comm.socketServerSetTimeout(timeout)
		client.reboot_core()
		i = frames
		t = os.clock()
		while i > 0 do
			emu.frameadvance()
			resp = comm.socketServerScreenShotResponse()
			if resp ~= 'ack' then
				i = -100
				print("Failed to get a proper response at " .. timeout .. ", increasing timeout")
			end
			i = i - 1
		end
		if i > -100 then
			print("Best timeout: " .. timeout .. " msecs")
			print("Best time:    " .. round((os.clock() - t ), 3) .. " secs")
			break
		end
	end
	
end

function test_http_response(frames)
	err = false
	print("Testing HTTP server response")
	client.reboot_core()
	i = frames
	
	while i > 0 do
		emu.frameadvance()
		resp = comm.httpTestGet()
		if resp ~= "<html><body><h1>hi!</h1></body></html>" then
			print("Failed to get correct HTTP get response")
			i = 0
			err = true
		end
		i = i - 1
	end
	if not err then
		print("HTTP GET looks fine: No errors occurred")
	end
	
	client.reboot_core()
	i = frames
	err = false
	while i > 0 do
		emu.frameadvance()
		resp = comm.httpPostScreenshot()
		if resp ~= "<html><body>OK</body></html>" then
			print("Failed to get correct HTTP post response")
			i = 0
			err = true
		end
		i = i - 1
	end
	if not err then
		print("HTTP POST looks fine: No errors occurred")
	end
end

frames = 100
baseline = get_baseline(frames)
--test_socket(frames)
test_mmf(frames)
test_http(frames)
print("#####################")
test_http_response(frames)
--test_socketresponse(frames)
print("---")

