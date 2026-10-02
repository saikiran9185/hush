-- Hush for DaVinci Resolve: split a clip's audio into layers, one per sound.
--
-- Select a clip, then Workspace > Scripts > Hush - Extract Layers. Hush finds the sounds in it
-- (horns, traffic, wind, music…) and puts each on its own audio track, in sync, named after the
-- sound; the voice stays on "Voice & rest". Mute a track to remove that sound. The original clip's
-- audio is switched off, not deleted. The clip is orange while Hush works.
--
-- Resolve Free can't run Python or show windows from scripts, so this hands the clip to hush.py
-- through a file Resolve writes itself (ExportMetadata) and waits for the layers.
-- Errors from this script are logged in
-- ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/logs/ResolveDebug.txt

local HOME = os.getenv("HOME")
local INBOX = HOME .. "/Library/Application Support/Hush/inbox/"
local STATUS = HOME .. "/Library/Application Support/Hush/status/"
local MEDIA = HOME .. "/Movies/Hush/"

local resolve = resolve or Resolve()
local project = resolve:GetProjectManager():GetCurrentProject()
local timeline = project and project:GetCurrentTimeline()
if not timeline then return end
local pool = project:GetMediaPool()
local fps = tonumber(tostring(timeline:GetSetting("timelineFrameRate")):match("^[%d%.]+")) or 24
local id = string.format("%d%04d", os.time(), math.random(0, 9999))

local function list(t) -- Resolve's Lua lists carry extra keys; keep only the objects
  local out = {}
  for _, v in pairs(t or {}) do if type(v) == "userdata" then out[#out + 1] = v end end
  return out
end

local function kind(item) return (item:GetTrackTypeAndIndex() or {})[1] end

local function marker(item, name, note)
  timeline:AddMarker(math.floor(item:GetStart() - timeline:GetStartFrame()), "Red", name, note, 1, "hush")
end

-- 1. Which clips: the selected ones (a selected video clip means its audio), else the one under the playhead.
local clips, seen = {}, {}
local function add(item)
  if kind(item) == "audio" and not seen[item:GetUniqueId()] then
    seen[item:GetUniqueId()] = true
    clips[#clips + 1] = item
  end
end
for _, item in ipairs(list(timeline:GetSelectedClips())) do
  if kind(item) == "video" then
    for _, linked in ipairs(list(item:GetLinkedItems())) do add(linked) end
  end
  add(item)
end
if #clips == 0 then
  local h, m, s, f = tostring(timeline:GetCurrentTimecode()):match("(%d+)[:;](%d+)[:;](%d+)[:;](%d+)")
  local frame = h and ((h * 60 + m) * 60 + s) * math.floor(fps + 0.5) + f
  for track = 1, frame and timeline:GetTrackCount("audio") or 0 do
    for _, item in ipairs(list(timeline:GetItemListInTrack("audio", track))) do
      if item:GetStart() <= frame and frame < item:GetEnd() then add(item) end
    end
    if #clips > 0 then break end
  end
end

-- 2. One request per clip: the media file (via ExportMetadata) + the part of it the clip uses.
local jobs, retimed = {}, false
for _, item in ipairs(clips) do
  local s, e = item:GetSourceStartTime(), item:GetSourceEndTime()
  if math.abs((e - s) - (item:GetEnd() - item:GetStart()) / fps) > 0.02 * (e - s) + 0.05 then
    retimed = true
  elseif item:GetMediaPoolItem() then
    jobs[#jobs + 1] = { item = item, s = s, e = e, layer = 1 }
  end
end
if #jobs == 0 then -- hush.py shows the message; the file name is the message
  return timeline:Export(INBOX .. id .. "-msg_" .. (retimed and "retimed" or "noclip") .. ".edl",
    resolve.EXPORT_EDL, resolve.EXPORT_NONE)
end
for k, job in ipairs(jobs) do
  local name = string.format("%s-%dof%d_s%.6f_e%.6f.csv", id, k, #jobs, job.s, job.e)
  if not pool:ExportMetadata(INBOX .. name, { job.item:GetMediaPoolItem() }) then
    return marker(job.item, "Hush is not installed", "Run install.sh in the Hush folder, then try again.")
  end
  job.color = job.item:GetClipColor()
  job.item:SetClipColor("Orange") -- "working on it"
end

local function restore(job)
  if job.color ~= "" then job.item:SetClipColor(job.color) else job.item:ClearClipColor() end
end

local deadline = os.time() + 20
while not bmd.fileexists(STATUS .. id .. ".ack") do
  if os.time() > deadline then
    for _, job in ipairs(jobs) do restore(job) end
    return marker(jobs[1].item, "Hush didn't start", "Run install.sh in the Hush folder again, then try again.")
  end
  bmd.wait(0.25)
end

-- 3. Put each layer on its own track as soon as it's ready.
local function hushBin()
  for _, folder in ipairs(list(pool:GetRootFolder():GetSubFolderList())) do
    if folder:GetName() == "Hush" then return folder end
  end
  return pool:AddSubFolder(pool:GetRootFolder(), "Hush")
end

local function trackFor(name, first, last) -- an audio track called `name` that's free here, else a new one
  for track = 1, timeline:GetTrackCount("audio") do
    local free = timeline:GetTrackName("audio", track) == name
    for _, other in ipairs(free and list(timeline:GetItemListInTrack("audio", track)) or {}) do
      if other:GetStart() < last and first < other:GetEnd() then free = false end
    end
    if free then return track end
  end
  timeline:AddTrack("audio", "stereo")
  timeline:SetTrackName("audio", timeline:GetTrackCount("audio"), name)
  return timeline:GetTrackCount("audio")
end

local function place(tag, job)
  local prefix, file = tag .. "-" .. job.layer .. "__", nil
  for _, path in pairs(resolve:GetMediaStorage():GetFileList(MEDIA:sub(1, -2)) or {}) do
    local base = tostring(path):match("[^/]+$") or ""
    if base:sub(1, #prefix) == prefix then file = base end
  end
  if not file then return end
  local name = file:sub(#prefix + 1):gsub("%.wav$", "")
  if project:GetCurrentTimeline():GetUniqueId() ~= timeline:GetUniqueId() then project:SetCurrentTimeline(timeline) end

  local folder = pool:GetCurrentFolder()
  pool:SetCurrentFolder(hushBin())
  local clip = list(pool:ImportMedia({ MEDIA .. file }))[1]
  pool:SetCurrentFolder(folder)
  clip:SetClipProperty("Clip Name", job.item:GetName() .. " – " .. name)

  local playhead = timeline:GetCurrentTimecode()
  local first, last = job.item:GetStart(), job.item:GetEnd()
  local placed = list(pool:AppendToTimeline({ { mediaPoolItem = clip, mediaType = 2, recordFrame = first,
    trackIndex = trackFor(name, first, last) } }))[1]
  bmd.wait(0.3) -- AppendToTimeline moves the playhead after it returns
  timeline:SetCurrentTimecode(playhead)
  local group = { placed }
  for _, linked in ipairs(list(job.item:GetLinkedItems())) do
    if kind(linked) == "video" then group[#group + 1] = linked end
  end
  if #group > 1 then timeline:SetClipsLinked(group, true) end
end

local left, errors = #jobs, {}
local function collect()
  for k, job in ipairs(jobs) do
    local tag = id .. "-" .. k
    while not job.finished and bmd.fileexists(STATUS .. tag .. "-" .. job.layer .. ".layer") do
      local ok, err = pcall(place, tag, job)
      if not ok then errors[#errors + 1] = tostring(err) end
      job.layer = job.layer + 1
    end
    local done = bmd.fileexists(STATUS .. tag .. ".done")
    if not job.finished and (done or bmd.fileexists(STATUS .. tag .. ".failed")) then
      if done and job.layer > 1 then job.item:SetClipEnabled(false) end -- the layers replace it
      job.finished, left = true, left - 1
      restore(job)
    end
  end
end

-- hush.py deletes the .ack when it finishes or gives up, so this never hangs.
deadline = os.time() + 3 * 3600
while left > 0 and os.time() < deadline and bmd.fileexists(STATUS .. id .. ".ack") do
  collect()
  bmd.wait(0.5)
end
collect()
for _, job in ipairs(jobs) do if not job.finished then restore(job) end end
if #errors > 0 then error("Hush: " .. table.concat(errors, " | ")) end
