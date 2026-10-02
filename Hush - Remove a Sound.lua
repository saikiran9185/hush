-- Hush for DaVinci Resolve: select a sound, name it, remove it.
--
--   1. Mark In and Out around the sound (keys I and O). No marks = the whole selected clip.
--   2. Workspace > Scripts > Hush - Remove a Sound. Type what the sound is, press Return.
--   3. The cleaned clip lands in sync on a "Hush" track; the original audio is switched off, not deleted.
--
-- Resolve Free can't run Python or show windows from scripts, so this hands the job to hush.py
-- through files Resolve writes itself (ExportMetadata) and waits for the result.
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

local function tell(code) -- hush.py shows the message; the file name is the message
  timeline:Export(INBOX .. id .. "-msg_" .. code .. ".edl", resolve.EXPORT_EDL, resolve.EXPORT_NONE)
end

local function marker(item, name, note)
  timeline:AddMarker(math.floor(item:GetStart() - timeline:GetStartFrame()), "Red", name, note, 1, "hush")
end

-- 1. The marked range in timeline frames (Out is inclusive), else the playhead.
local marks = timeline:GetMarkInOut() or {}
local m = marks.audio or marks.video or {}
local from, to = m["in"], m.out and m.out + 1
if from and from < timeline:GetStartFrame() then -- marks relative to the timeline start
  from, to = from + timeline:GetStartFrame(), to + timeline:GetStartFrame()
end
if not from then
  local h, mi, s, f = tostring(timeline:GetCurrentTimecode()):match("(%d+)[:;](%d+)[:;](%d+)[:;](%d+)")
  from = h and ((h * 60 + mi) * 60 + s) * math.floor(fps + 0.5) + f
  to = from and from + 1
end

-- 2. Which clips: the selected ones, else those under the marks on the first audio track that has any.
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
for track = 1, (#clips == 0 and from) and timeline:GetTrackCount("audio") or 0 do
  for _, item in ipairs(list(timeline:GetItemListInTrack("audio", track))) do
    if item:GetStart() < to and from < item:GetEnd() then add(item) end
  end
  if #clips > 0 then break end
end

-- 3. One request per clip: the media file (via ExportMetadata) + clip and marked range in source seconds.
local jobs, retimed = {}, false
for _, item in ipairs(clips) do
  local s, e, first, last = item:GetSourceStartTime(), item:GetSourceEndTime(), item:GetStart(), item:GetEnd()
  local a, b = first, last
  if m["in"] then a, b = math.max(first, from), math.min(last, to) end
  if math.abs((e - s) - (last - first) / fps) > 0.02 * (e - s) + 0.05 then
    retimed = true
  elseif item:GetMediaPoolItem() and b > a then
    jobs[#jobs + 1] = { item = item, s = s, e = e, a = s + (a - first) / fps, b = s + (b - first) / fps }
  end
end
if #jobs == 0 then return tell(retimed and "retimed" or "noclip") end

for k, job in ipairs(jobs) do
  local name = string.format("%s-%dof%d_s%.6f_e%.6f_a%.6f_b%.6f.csv", id, k, #jobs, job.s, job.e, job.a, job.b)
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

-- 4. Put each cleaned clip back in sync as it finishes.
local function hushBin()
  for _, folder in ipairs(list(pool:GetRootFolder():GetSubFolderList())) do
    if folder:GetName() == "Hush" then return folder end
  end
  return pool:AddSubFolder(pool:GetRootFolder(), "Hush")
end

local function hushTrack(first, last) -- a "Hush" audio track that's free here, else a new one
  for track = 1, timeline:GetTrackCount("audio") do
    local free = timeline:GetTrackName("audio", track) == "Hush"
    for _, other in ipairs(free and list(timeline:GetItemListInTrack("audio", track)) or {}) do
      if other:GetStart() < last and first < other:GetEnd() then free = false end
    end
    if free then return track end
  end
  timeline:AddTrack("audio", "stereo")
  timeline:SetTrackName("audio", timeline:GetTrackCount("audio"), "Hush")
  return timeline:GetTrackCount("audio")
end

local function place(k, job)
  local prefix, file = id .. "-" .. k .. "__", nil
  for _, path in pairs(resolve:GetMediaStorage():GetFileList(MEDIA:sub(1, -2)) or {}) do
    local base = tostring(path):match("[^/]+$") or ""
    if base:sub(1, #prefix) == prefix then file = base end
  end
  if not file then return end
  if project:GetCurrentTimeline():GetUniqueId() ~= timeline:GetUniqueId() then project:SetCurrentTimeline(timeline) end

  local folder = pool:GetCurrentFolder()
  pool:SetCurrentFolder(hushBin())
  local clip = list(pool:ImportMedia({ MEDIA .. file }))[1]
  pool:SetCurrentFolder(folder)
  clip:SetClipProperty("Clip Name", (file:sub(#prefix + 1):gsub("%.wav$", "")))

  local playhead = timeline:GetCurrentTimecode()
  local placed = list(pool:AppendToTimeline({ { mediaPoolItem = clip, mediaType = 2, recordFrame = job.item:GetStart(),
    trackIndex = hushTrack(job.item:GetStart(), job.item:GetEnd()) } }))[1]
  bmd.wait(0.3) -- AppendToTimeline moves the playhead after it returns
  timeline:SetCurrentTimecode(playhead)
  local group = { placed }
  for _, linked in ipairs(list(job.item:GetLinkedItems())) do
    if kind(linked) == "video" then group[#group + 1] = linked end
  end
  if #group > 1 then timeline:SetClipsLinked(group, true) end
  job.item:SetClipEnabled(false)
end

local left, errors = #jobs, {}
local function collect()
  for k, job in ipairs(jobs) do
    local base = STATUS .. id .. "-" .. k
    if not job.finished and (bmd.fileexists(base .. ".done") or bmd.fileexists(base .. ".failed")) then
      if bmd.fileexists(base .. ".done") then
        local ok, err = pcall(place, k, job)
        if not ok then errors[#errors + 1] = tostring(err) end
      end
      job.finished, left = true, left - 1
      restore(job)
    end
  end
end

-- hush.py deletes the .ack when it finishes or gives up (cancelled, crashed), so this never hangs.
deadline = os.time() + 3 * 3600
while left > 0 and os.time() < deadline and bmd.fileexists(STATUS .. id .. ".ack") do
  collect()
  bmd.wait(0.5)
end
collect()
for _, job in ipairs(jobs) do if not job.finished then restore(job) end end
if #errors > 0 then error("Hush: " .. table.concat(errors, " | ")) end
