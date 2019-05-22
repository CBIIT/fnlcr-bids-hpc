 -- This version uses OpenMPI v3.1.2 as a Biowulf module
 -- This used to be called 2019-04-04
 -- Note that the versions of the checkouts in this version are from 2019-05-06 and for Supervisor from 2019-05-22

whatis("Version: openmpi")
whatis("URL: https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle")
whatis("Description: Open source software for scalable hyperparameter optimization")

local app         = "candle"
local version     = "openmpi"
local base = "/data/BIDS-HPC/public/software/distributions/candle/openmpi"

setenv("CANDLE", base) -- used by submit_candle_job.sh, run_without_candle.sh, and copy_candle_template.sh
 -- append_path("PATH", pathJoin(base,"scripts")) -- used only in order to find the copy_candle_template script
append_path("PATH", pathJoin(base,"Supervisor/templates/scripts")) -- used only in order to find the copy_candle_template script
setenv("SITE", "biowulf") -- used by submit_candle_job.sh
setenv("TURBINE_HOME", pathJoin(base,"builds/swift-t-install/turbine"))
setenv("MODULES_FOR_BUILD", "python/3.6")

if (mode() == "load") then
    LmodMessage("[+] Loading  ", app, version, " ...")
end
if (mode() == "unload") then
    LmodMessage("[-] Unloading ", app, version, " ...")
end
