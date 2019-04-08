
local app         = "candle"
local version     = "1.0"
local base = "/data/BIDS-HPC/public/software"

prepend_path("PATH", pathJoin(base,"checkouts","Candle"))
setenv("CANDLE_TEST","/data/classes/candle")

if (mode() == "load") then
    LmodMessage("[+] Loading  ", app, version, " ...")
end
if (mode() == "unload") then
    LmodMessage("[-] Unloading ", app, version, " ...")
end

