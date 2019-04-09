local app         = "candle"
local version     = "2019-04-04"
local base = "/data/BIDS-HPC/public/candle"

setenv("CANDLE", base) -- used by submit_candle_job.sh, run_without_candle.sh, and copy_candle_template.sh
append_path("PATH", pathJoin(base,"bin")) -- used only in order to find the copy_candle_template script
setenv("SITE", "biowulf") -- used by submit_candle_job.sh

if (mode() == "load") then
    LmodMessage("[+] Loading  ", app, version, " ...")
end
if (mode() == "unload") then
    LmodMessage("[-] Unloading ", app, version, " ...")
end