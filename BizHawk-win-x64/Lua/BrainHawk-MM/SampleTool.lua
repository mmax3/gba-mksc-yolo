require("./BHClient")

c = BHClient:new("http://127.0.0.1:1337")
c:initialize()
--client.setwindowsize(3)

colors = {
    [1] = "#FF0000", -- Red
    [2] = "#00FF00", -- Green
    [3] = "#0000FF", -- Blue   
    [4] = "#FFFF00", -- Yellow
    [5] = "#FF00FF", -- Magenta
    [6] = "#00FFFF", -- Cyan
    [7] = "#FFFFFF", -- White
    [8] = "#000000", -- Black
    [9] = "#808080", -- Gray
    [10] = "#FFA500", -- Orange
}

labels = c:read_classes("../../../Code/classes.txt")

---local canvas = gui.createcanvas(480, 320);

while true do
	t = os.clock()
    --
    -- Perform actions from server
    --
    c:useControls()   -- Set the controls to use until the next frame
    c:advanceFrame()  -- Advance a single frame
    --
	boxes={{}}
	--
    -- Retrieve feedback from server
    --
    -- Is it time to communicate with the server?
    if c:timeToUpdate() then
		-- Save a screenshot on the server
        c:saveScreenshot()
        -- Build a list of statements (to send in one request to server)
        local statements = {
            --c:setStatement("x", 512, "INT"),  -- Set x = 512 (as a Python Int). No return
            --c:getStatement("y"),              -- Get value for y
			--c:getStatement("z"),              -- Get value for z
			c:getStatement("boxes"),
			c:getStatement("scores"),
			c:getStatement("class_ids"),
            c:updateStatement(),              -- Call server's update(). No return
            c:setControlsStatement(),         -- Returns controls from server
			c:getControlsStatement(),         -- Sends controls to server
            c:checkRestartStatement(),        -- Returns whether emulator should reset
            c:checkExitStatement()            -- Returns whether client should exit
        }
        -- Compiled Message:
        -- SET x Int 512;
        -- GET y;
		-- GET z;
        -- UPDATE;
        -- GET controls;
        -- SET controls;
        -- GET restart;
        -- GET exit
		
        -- Send statements, grab results
        --local y_response, z_response, controls_response, restart_response, exit_response = c:sendList(statements)
		local boxes_response, scores_response, cls_response, controls_response, restart_response, exit_response = c:sendList(statements)
        -- Send results to the appropriate functions
        --local yType, y = c:get(y_response)
		--local zType, z = c:get(z_response)
		boxesType, boxes = c:get(boxes_response)
		scoresType, scores = c:get(scores_response)
		clsType, cls = c:get(cls_response)
		c:setControls(controls_response)
        c:checkRestart(restart_response)
		--[[
        -- Note: This will drastically slow learning speed
		if not (y_response == "None") then 
			console.writeline("y: " .. yType .. " " .. y)
		end
		if not (z_response == "None") then 
			console.writeline("z: " .. zType .. " " .. z)
		end
		]]
		if not (boxes == nil) then
			--print(boxes[1])
			--console.writeline(boxes)
			for i, row in ipairs(boxes) do
				--print(row)
				--if not (row == nil) then
				local i_str = tostring(i)
				local class = cls[i_str]
				local color = colors[class]
				local label = labels[class+1]
				gui.drawBox(row[1],row[2],row[3],row[4],color)
				gui.drawString(row[1],row[2]-10,label .. " " .. scores[i_str] .. "%",color,nil,10)
				--gui.drawString(row[3],row[2]-10,label,color,nil,10)
				--end
			end
		else
			gui.clearGraphics()
		end
		
        -- Did the server tell us to exit?
        if c:checkExit(exit_response) then break end
    end
	gui.drawString(0,0,math.floor(1/(os.clock()-t)),"#FFFFFF",nil,10)	
end