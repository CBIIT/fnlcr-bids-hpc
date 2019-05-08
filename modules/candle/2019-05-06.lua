 -- This version uses MPICH v3.3, built by Andrew

whatis("Version: 2019-05-06")
whatis("URL: https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle")
whatis("Description: Open source software for scalable hyperparameter optimization")

local app         = "candle"
local version     = "2019-05-06"
local base = "/data/BIDS-HPC/public/software/distributions/candle/2019-05-06"

setenv("CANDLE", base) -- used by submit_candle_job.sh, run_without_candle.sh, and copy_candle_template.sh
append_path("PATH", pathJoin(base,"scripts")) -- used only in order to find the copy_candle_template script
setenv("SITE", "biowulf") -- used by submit_candle_job.sh

if (mode() == "load") then
    LmodMessage("[+] Loading  ", app, version, " ...")
end
if (mode() == "unload") then
    LmodMessage("[-] Unloading ", app, version, " ...")
end