"""
core/knowledge_base.py
======================
Deterministic, offline **Cat machinery knowledge base** used by CAT-Pal in ``offline``
mode (the default on an edge node with no internet).

Why a knowledge base at all?
----------------------------
1. **Availability** - a haul road with no uplink still gets useful answers.
2. **Determinism** - the same question yields the same answer, which is testable.
3. **Safety** - an LLM cannot hallucinate a torque value or a bypass procedure when the
   answer is selected from reviewed, static text that always defers to the machine's
   Operation & Maintenance Manual (OMM).

Retrieval is a transparent lexical scorer (keyword phrase weight + title overlap), not an
embedding model: it runs in microseconds on a 2 GB edge box and its behaviour is fully
explainable in an audit log.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Stop-words removed before scoring (keeps the allow-list check meaningful)
# --------------------------------------------------------------------------------------
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been being do does did doing have has had having
    i me my we our you your it its this that these those of in on at to for with
    from by about into over after before and or but if then than as so such no not
    can could should would will shall may might must please tell show give need want
    how what when where which who whom why what's whats why's
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/\.]{1,24}")


@dataclass(frozen=True)
class KnowledgeEntry:
    """One reviewed answer topic."""

    entry_id: str
    title: str
    keywords: tuple[str, ...]
    body: str
    category: str = "machine"
    caution: str = ""

    def render(self) -> str:
        """Markdown rendering (static text only - safe for ``st.markdown``)."""
        parts = [f"**{self.title}**", "", self.body.strip()]
        if self.caution:
            parts += ["", f"⚠️ {self.caution.strip()}"]
        return "\n".join(parts)


# --------------------------------------------------------------------------------------
# The knowledge base
# --------------------------------------------------------------------------------------
KB_ENTRIES: tuple[KnowledgeEntry, ...] = (
    KnowledgeEntry(
        entry_id="ENGINE_OVER_REV",
        title="Engine over-speed (over-rev) event",
        keywords=("over rev", "over-rev", "overrev", "engine rpm high", "rpm too high",
                  "engine speed", "governor", "high rpm", "engine overspeed"),
        category="engine",
        body=(
            "A short over-speed blip is normally the governor catching a sudden unload rather "
            "than a fault. Treat it as actionable when it repeats or lasts:\n\n"
            "- Park on level ground, set the parking brake, let the engine idle for 2-3 minutes "
            "to shed heat before shutdown.\n"
            "- Check for stored fault codes on the machine monitor and record them in the shift log.\n"
            "- Common mechanical causes: air in the fuel system, a sticking injector rack, a "
            "restricted air filter, or an incorrect (field-modified) high-idle setting.\n"
            "- Do not adjust the governor or injection pump in the field - that is dealer-level work."
        ),
        caution="Repeated over-speed trips mean stop-work until a dealer technician inspects the engine.",
    ),
    KnowledgeEntry(
        entry_id="HIGH_IDLE_FUEL",
        title="High idle and fuel waste",
        keywords=("high idle", "idle", "idling", "fuel waste", "excess fuel", "fuel consumption",
                  "fuel economy", "burn rate", "litres per hour", "fuel burn", "eco mode"),
        category="productivity",
        body=(
            "Idling burns fuel with zero tonnes moved. On a 320-class excavator expect roughly "
            "6-9 L/h at low idle versus 28-40 L/h while digging; that idle fuel is pure loss.\n\n"
            "- Use **auto-idle / auto engine shutdown** so the engine drops to low idle after the "
            "configured delay (typically 3-10 s) whenever no hydraulic demand is present.\n"
            "- During truck spotting, cut the engine rather than idling through the wait.\n"
            "- Track L/t (litres per tonne) and idle % each shift - the anomaly log flags sustained "
            "idle windows longer than five minutes automatically."
        ),
        caution="Never disable auto-idle or auto-shutdown to 'save time' - it is a fuel and emissions control.",
    ),
    KnowledgeEntry(
        entry_id="HYDRAULIC_PRESSURE",
        title="Hydraulic pressure: normal range and faults",
        keywords=("hydraulic pressure", "hydraulic psi", "system pressure", "relief valve",
                  "main relief", "over pressure", "under pressure", "hydraulic pump",
                  "implement pressure", "pressure too low", "pressure too high"),
        category="hydraulics",
        body=(
            "On a warm machine, main implement pressure at relief typically sits in the "
            "3 500-5 000 psi band; low-pressure standby is a few hundred psi and work-port "
            "pressure while digging should climb smoothly.\n\n"
            "- **Erratic or low pressure while digging**: check hydraulic oil level, clogged "
            "suction strainer, aeration (foamy oil), or a worn pump.\n"
            "- **Pressure that stays high at idle**: a spool stuck in stroke or a failed relief - "
            "stop work, the oil is being turned into heat.\n"
            "- Always verify with the pressure gauge from the OMM's Testing & Adjusting section; "
            "do not shim relief valves to 'get more power'."
        ),
        caution="Hydraulic oil at working pressure can inject through skin. Never check for leaks with bare hands - use cardboard.",
    ),
    KnowledgeEntry(
        entry_id="HYDRAULIC_OVERHEAT",
        title="Hydraulic oil over-temperature",
        keywords=("hydraulic temperature", "hydraulic oil temp", "oil overheat", "hydraulic overtemp",
                  "hydraulic hot", "hydraulic cooler", "hydraulic oil temperature"),
        category="hydraulics",
        body=(
            "Sustained hydraulic oil above ~95 °C thins the oil, degrades seals and shortens pump life.\n\n"
            "- Clean the hydraulic cooler core with low-pressure water or dry compressed air "
            "(from the fan side, with the machine off and cool).\n"
            "- Check the cooler bypass valve, thermostat and the oil level (low oil = less dwell "
            "time in the cooler).\n"
            "- Reduce the duty cycle during high ambient temperature: shorter continuous digging "
            "with a cool-down idle every hour."
        ),
        caution="Never open a hydraulic tank or loosen a fitting while the oil is hot - pressurised hot oil causes severe burns.",
    ),
    KnowledgeEntry(
        entry_id="COOLANT_TEMP",
        title="Engine coolant over-temperature",
        keywords=("coolant temperature", "coolant temp", "engine overheat", "overheating",
                  "coolant", "radiator", "temperature warning", "engine hot", "thermostat"),
        category="engine",
        body=(
            "- Stop working at the first over-temperature warning, move to a safe area and idle "
            "at 1 000-1 200 rpm (not full speed) so the coolant keeps circulating.\n"
            "- Do not remove the radiator cap while hot. Wait until the engine and radiator are "
            "cool enough to touch, then release the cap slowly to the first stop.\n"
            "- Typical causes: low coolant, plugged/blocked radiator fins, failed thermostat, "
            "slipping fan belt, or a failed water pump.\n"
            "- Top up with the coolant mix specified in the OMM; straight water changes the "
            "freeze/boil point and the additive package."
        ),
        caution="A hot pressurised cooling system can scald. Allow 30+ minutes of cool-down before opening anything.",
    ),
    KnowledgeEntry(
        entry_id="OIL_PRESSURE",
        title="Low engine oil pressure",
        keywords=("oil pressure", "low oil pressure", "oil light", "engine oil pressure",
                  "oil pump", "oil pressure low"),
        category="engine",
        body=(
            "Low engine oil pressure is a stop-immediately condition.\n\n"
            "- Shut down, park safely, and check oil level on the dipstick with the machine level.\n"
            "- If the level is correct, do not restart: a failed oil pump or a blocked pickup screen "
            "will destroy bearings in minutes.\n"
            "- Look for an external leak, a loose oil filter, or a faulty pressure sender before "
            "assuming the worst - a sender fault is recorded as a sensor fault in the telemetry log."
        ),
        caution="Never continue working with a low oil pressure warning. Engine seizure risk is immediate.",
    ),
    KnowledgeEntry(
        entry_id="SEATBELT_INTERLOCK",
        title="Seatbelt interlock and operator restraint",
        keywords=("seatbelt", "seat belt", "safety belt", "seatbelt interlock", "belt latched",
                  "belt unlatched", "operator restraint", "belt alarm"),
        category="safety",
        body=(
            "The seatbelt interlock is a life-safety device, not an inconvenience.\n\n"
            "- Latch the belt **before** raising the lock lever or starting the engine, and keep it "
            "latched for every travel movement, even a 10 m reposition.\n"
            "- If the indicator shows unlatched while moving, stop the machine, latch the belt, and "
            "clear the alert before continuing.\n"
            "- A belt that will not latch, or an interlock that will not clear when latched, is a "
            "defect: tag the machine and raise a service request."
        ),
        caution="Never defeat or bypass the seatbelt interlock. Do not place the belt behind the operator or over the seat frame.",
    ),
    KnowledgeEntry(
        entry_id="PROXIMITY_RADAR",
        title="Proximity detection radar zones",
        keywords=("proximity", "radar", "proximity warning", "object detection", "collision warning",
                  "spotter", "swing radius", "people detection", "camera system", "detection zone"),
        category="safety",
        body=(
            "The detection system uses three zones around the machine:\n\n"
            "- **CAUTION (≈15 m)** - pedestrian or vehicle detected; maintain awareness and keep "
            "the bucket low.\n"
            "- **WARNING (≈10 m)** - slow the swing and travel, sound the horn, confirm the spotter.\n"
            "- **STOP (≈5 m)** - stop the swing/travel immediately. Hold until the zone clears and "
            "you have eye contact or a radio call with the person.\n\n"
            "The system is an aid, not a substitute for mirrors, cameras, a spotter or a "
            "traffic-management plan. Re-check sensor cleanliness each pre-start."
        ),
        caution="Never mute, cover or disable a proximity sensor to stop alarms. A silenced alarm hides a person, not an obstacle.",
    ),
    KnowledgeEntry(
        entry_id="PRE_START",
        title="Daily pre-start / walk-around inspection",
        keywords=("pre-start", "prestart", "walk around", "walkaround", "daily check",
                  "daily inspection", "start of shift", "checks before", "fluid check"),
        category="maintenance",
        body=(
            "Run the pre-start on a level surface with the engine off, implements lowered and "
            "pressure released (10-15 minutes):\n\n"
            "1. Walk-around: leaks, loose/missing hardware, cracked glass, damaged guards, mud build-up.\n"
            "2. Fluid levels: engine oil, coolant, hydraulic oil, final drives, fuel/water separator drain.\n"
            "3. Undercarriage: track tension and sag, sprocket/idler wear, missing or loose shoes.\n"
            "4. Bucket and GET: tooth and adapter wear, side-cutter bolts, cracked lip.\n"
            "5. Cab: seatbelt and interlock function, mirrors, camera/radar cleanliness, ROPS/FOPS, fire "
            "extinguisher charge, horn and travel alarm.\n"
            "6. Monitor: record any active fault codes before moving the machine."
        ),
        caution="Report defects instead of 'running to failure'. Log every issue in the shift handover.",
    ),
    KnowledgeEntry(
        entry_id="LOTO",
        title="Lockout / tagout (LOTO) and safe shutdown",
        keywords=("lock out tag out", "lockout", "tagout", "loto", "isolation", "safe shutdown",
                  "maintenance safety", "de-energise", "de-energize", "hydraulic lock", "stored energy"),
        category="safety",
        body=(
            "No work on or under a machine until it is isolated:\n\n"
            "1. Move to a level, firm, safe area; lower all implements to the ground.\n"
            "2. Apply the parking brake, shift to neutral, shut down the engine and remove the key.\n"
            "3. Isolate and lock the battery disconnect, and any other energy source (quick coupler, "
            "accumulator, attachment, electric drive on hybrid/electric machines).\n"
            "4. **Release stored energy**: cycle the controls to bleed residual hydraulic pressure, "
            "block or lower suspended components, and allow hot surfaces to cool.\n"
            "5. Attach your personal lock and tag; only the person who applied the lock removes it.\n"
            "6. For any hydraulic lock, interlock or limit that will not release, raise a service "
            "request - field bypasses are never acceptable."
        ),
        caution="Stored hydraulic and accumulator pressure can move a cylinder with the engine off. Always bleed and block before working.",
    ),
    KnowledgeEntry(
        entry_id="HAUL_CYCLE",
        title="Haul-cycle productivity",
        keywords=("haul cycle", "cycle time", "cycle", "productivity", "tons per hour", "tph",
                  "tonnage", "bucket fill", "truck spotting", "match factor", "payload", "target"),
        category="productivity",
        body=(
            "Cycle time is the sum of dig, swing (loaded), dump/spot and return swing. Typical 320-class "
            "excavator cycle times on a well-matched truck fleet run 22-32 s; anything above 40 s means a "
            "bottleneck somewhere in the chain.\n\n"
            "- **Bucket fill**: aim for a heaped, escaping-free load. Two passes with a full bucket beat "
            "three with a half bucket - the swing time is the same either way.\n"
            "- **Swing discipline**: stop the swing with the bucket above the truck body centre, dump, and "
            "start the return swing before the truck moves off.\n"
            "- **Truck spotting**: the biggest hidden loss. Keep the truck spotting area clean and level, "
            "and use a fixed dump target so the operator does not chase the body.\n"
            "- Track tons/hour against the shift plan each hour, not at the end of the shift."
        ),
        caution="Never exceed rated payload to chase a tonnage target - overloading is a safety and warranty risk.",
    ),
    KnowledgeEntry(
        entry_id="UNDERCARRIAGE",
        title="Undercarriage and track maintenance",
        keywords=("undercarriage", "track", "tracks", "track tension", "track shoe", "sprocket",
                  "idler", "roller", "track sag", "undercarriage wear", "shoe bolts"),
        category="maintenance",
        body=(
            "Undercarriage is the single largest maintenance cost on a crawler machine.\n\n"
            "- Check track sag per the OMM (measured with the machine raised correctly - never crawl "
            "under a machine supported only by its own implements).\n"
            "- Tighter tracks are not better: excessive tension accelerates pin, bushing and sprocket wear.\n"
            "- Look for missing/loose shoe bolts, oil leaks at rollers and idlers, and scalloped sprocket teeth.\n"
            "- Avoid high-speed travel on rock and minimise spot turning on concrete - it eats both tracks "
            "and floor."
        ),
        caution="Never work under an implement-supported machine. Use certified stands in the LOTO procedure.",
    ),
    KnowledgeEntry(
        entry_id="GET_BUCKET",
        title="Bucket teeth and ground-engaging tools (GET)",
        keywords=("bucket", "teeth", "tooth", "adapter", "side cutter", "get", "ground engaging",
                  "wear plate", "cutting edge", "bucket lip", "quick coupler", "coupler"),
        category="maintenance",
        body=(
            "- Inspect teeth and adapters at every pre-start: worn tips blunt penetration and spike "
            "digging force and fuel burn.\n\n"
            "- Rotate or replace GET when wear reaches the OMM limit - a worn adapter takes the bucket lip "
            "with it and turns a cheap job into a weld-repair job.\n"
            "- Keep the quick-coupler locking system clean and test the engagement in a safe area each shift.\n"
            "- Crack propagation from the lip or side plate is a stop-work defect."
        ),
        caution="Never use the bucket as a lifting device unless the machine and rigging are rated and the lift plan is approved.",
    ),
    KnowledgeEntry(
        entry_id="FAULT_CODES",
        title="Reading and reporting fault codes",
        keywords=("fault code", "error code", "diagnostic", "diagnostic code", "code list",
                  "dash warning", "monitor message", "eid", "cid", "troubleshoot"),
        category="diagnostics",
        body=(
            "- Fault codes are shown on the monitor with a component identifier (CID) and an event "
            "identifier (EID), for example a high coolant temperature event.\n\n"
            "- Record the code, the machine hours, the conditions (load, temperature, speed) and whether "
            "it is active or logged only, then pass it to maintenance.\n"
            "- A logged-only code from months ago is informational; an active code that de-rates the "
            "machine is a stop-work decision - never derate-clear your way through a shift.\n"
            "- Cat ET / dealer software is required to clear certain codes and to see freeze-frame data."
        ),
        caution="Never clear active fault codes to get the machine moving again. Fix the cause, then clear.",
    ),
    KnowledgeEntry(
        entry_id="FUEL_DEF_FILTERS",
        title="Fuel, DEF and filter servicing",
        keywords=("fuel", "diesel", "def", "adblue", "diesel exhaust fluid", "fuel filter",
                  "water separator", "filter", "service interval", "gregreasing", "lubrication",
                  "engine oil change", "oil sample", "scheduled maintenance"),
        category="maintenance",
        body=(
            "- **Fuel**: drain the water separator daily; refuel at the end of the shift to reduce tank "
            "condensation. Use clean, low-sulphur diesel - dirt and water are the top cause of injector damage.\n"
            "- **DEF**: use only ISO 22241 fluid from a sealed container. Contaminated DEF causes derate "
            "events and expensive aftertreatment repairs; never water it down.\n"
            "- **Filters**: replace at the OMM interval, not 'when it looks dirty'. Pre-fill fuel filters "
            "with clean fuel using the prescribed procedure.\n"
            "- **Oil sampling**: take samples warm and mid-interval to catch wear metals trends early.\n"
            "- **Greasing**: follow the OMM grease chart - under-greasing pins and bushings is the cheapest "
            "way to destroy a machine."
        ),
        caution="Do not open the fuel system downstream of the filters, and never clean DEF with anything but clean water.",
    ),
    KnowledgeEntry(
        entry_id="TRAVEL_SLOPES",
        title="Travel, slopes and stability",
        keywords=("travel", "tramming", "slope", "gradient", "stability", "tipping", "ramp",
                  "travel on slope", "hill", "bank", "bench"),
        category="operation",
        body=(
            "- Travel with the bucket low (roughly 200-300 mm above ground) and the idlers (not sprockets) "
            "facing downhill.\n\n"
            "- On slopes, keep the machine as level as practicable: travel up/down the fall line (90° to the "
            "contour), never across it. If it must be crossed, keep the slope within the OMM limit and the "
            "travel speed at walking pace.\n"
            "- Never travel with a raised bucket or with the upper structure swung to the side - that is a "
            "stability loss, not a time saving.\n"
            "- Keep travel distances on a bench short; walking a crawler machine is expensive and slow."
        ),
        caution="A machine can tip faster than an operator can react. If it feels unstable, stop, lower everything, and re-plan the route.",
    ),
    KnowledgeEntry(
        entry_id="PARKING_SHUTDOWN",
        title="Correct parking and shutdown",
        keywords=("shutdown", "shut down", "parking", "park the machine", "secure machine",
                  "end of shift", "stop procedure", "idle down", "cooldown", "cooldown period"),
        category="operation",
        body=(
            "1. Park on level, firm ground clear of traffic and overburden.\n"
            "2. Lower the bucket and all implements to the ground; retract the bucket cylinder.\n"
            "3. Idle for 2-3 minutes (turbo and hydraulic heat soak), then shut down.\n"
            "4. Apply the parking brake, remove the key, set the lock lever to LOCKED.\n"
            "5. Walk around: leaks, damaged GET, hot spots, panel security.\n"
            "6. Record hours, fuel added, faults and the shift summary for the handover."
        ),
        caution="Never shut a loaded turbo engine down from full load - heat soak without oil flow cokes the turbo bearings.",
    ),
    KnowledgeEntry(
        entry_id="SITE_TRAFFIC",
        title="Jobsite traffic management and utility strikes",
        keywords=("traffic", "traffic management", "haul road", "site rules", "utility",
                  "buried services", "underground cable", "gas line", "excavation permit",
                  "pipe", "clearance", "hand dig", "locate"),
        category="safety",
        body=(
            "- Agree the traffic-management plan before the first bucket: haul-road directions, one-way "
            "loops, spotting areas, pedestrian exclusion zones and radio channels.\n\n"
            "- Physically barricade the swing radius where the machine works next to haul roads or footpaths.\n"
            "- Dig with a spotter and hand tools within the tolerance band of any marked service (buried "
            "cable, fibre, gas, water, sewer). Call for a utilities locate before you break ground - never "
            "assume a line's depth or alignment.\n"
            "- If you contact a service, stop, evacuate the immediate area, do not touch the machine, and "
            "call the emergency number on the permit."
        ),
        caution="A gas strike or live-cable contact is an emergency: stop, evacuate upwind, forbid re-entry and call the utility.",
    ),
    KnowledgeEntry(
        entry_id="COLD_START",
        title="Cold start and warm-up",
        keywords=("cold start", "cold weather", "warm up", "warmup", "starting procedure",
                  "ether", "block heater", "winter", "starting the engine", "battery low"),
        category="engine",
        body=(
            "- On cold starts, use the machine's starting aid (grid heater / block heater) rather than "
            "repeated cranking. Never use ether on an engine with an intake preheat system.\n\n"
            "- Crank in short bursts (max ~30 s), with a pause between attempts; continuous cranking kills "
            "the starter and floods the cylinders.\n"
            "- Warm up at low idle for several minutes, then exercise each hydraulic function slowly at low "
            "engine speed to bring the oil up to operating temperature before full duty.\n"
            "- If the battery is weak, jump-start with the correct procedure (parallel battery, engine off, "
            "correct polarity) or swap the battery - never bypass an isolation device to start."
        ),
        caution="Never start a machine with a person near the swing radius, implements or the articulation area.",
    ),
    KnowledgeEntry(
        entry_id="EMISSIONS_TIER",
        title="Emissions systems (Tier 4 Final / Stage V) and derate",
        keywords=("emissions", "tier 4", "stage v", "dpf", "scr", "egr", "aftertreatment",
                  "regeneration", "regen", "derate", "particulate filter", "nox"),
        category="engine",
        body=(
            "- A parked regeneration raises exhaust temperature far above normal: only start one in a "
            "permitted, clear area away from combustibles, free of pedestrians, and never in a "
            "confined space.\n\n"
            "- Keep the DEF tank above roughly one quarter - low DEF triggers derate warnings before low "
            "quality DEF would.\n"
            "- Fuel quality drives aftertreatment life: high sulphur diesel poisons the catalyst and is not "
            "covered by warranty.\n"
            "- Follow the OMM service interval for the DPF ash service; a clogged DPF shows up first as "
            "power loss and rising fuel burn."
        ),
        caution="Never park a regenerating machine over dry grass or on a fuel spill - the exhaust temperature can ignite debris.",
    ),
    KnowledgeEntry(
        entry_id="OPERATOR_FATIGUE",
        title="Operator fatigue and shift handover",
        keywords=("fatigue", "tired", "rest break", "handover", "shift handover", "micro sleep",
                  "alert", "operator wellness", "break", "shift log"),
        category="safety",
        body=(
            "- Fatigue is a recognised cause of machine damage and fatalities. Take the scheduled breaks, "
            "stand up, hydrate and get out of the cab during long idle waits.\n\n"
            "- At handover, walk the machine with the next operator and cover: active fault codes, fluid "
            "top-ups, damaged GET, fluid leaks, ground conditions, and any near-miss from the shift.\n"
            "- Log near-misses - they are free lessons. If you feel unable to work safely, stop and tell "
            "your supervisor; that is the correct call and it is supported."
        ),
        caution="Never continue operating while fighting sleep. Stop in a safe area and use the radio.",
    ),
    KnowledgeEntry(
        entry_id="TYRES_AND_TRAVEL_GEAR",
        title="Tyres (wheel loaders) and travel gear",
        keywords=("tyre", "tyres", "tire", "tires", "tire pressure", "wheel loader", "load and carry",
                  "steering", "brake", "brake test", "parking brake", "transmission"),
        category="operation",
        body=(
            "- Check tyre pressure weekly and after any tyre event: under-inflation overheats the "
            "carcass, over-inflation cuts traction and increases cutting.\n\n"
            "- On wheel loaders, load-and-carry stresses the tyres fastest - keep travel speeds moderate, "
            "loads low, and avoid turning with a full bucket at speed.\n"
            "- Test the service brake and parking brake at the start of every shift in a clear area; "
            "brake performance is one of the highest-consequence checks on the machine.\n"
            "- Inflation or tyre work: deflate before removing wheel nuts, use a safety cage, and stand "
            "clear of the wheel plane."
        ),
        caution="A tyre explosion kills. Never approach a suspected damaged tyre from the sidewall plane - use a remote gauge.",
    ),
)

KB_BY_ID: dict[str, KnowledgeEntry] = {entry.entry_id: entry for entry in KB_ENTRIES}


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------
@dataclass
class KnowledgeHit:
    """A scored retrieval result."""

    entry: KnowledgeEntry
    score: float = 0.0
    matched: tuple[str, ...] = field(default_factory=tuple)


def tokenize(text: str) -> list[str]:
    """Lower-case, NFKC-normalise and split text into scoring tokens (stop-words removed)."""
    normalised = unicodedata.normalize("NFKC", text or "").lower()
    return [tok for tok in _TOKEN_RE.findall(normalised) if tok not in STOPWORDS and len(tok) > 1]


def score_entries(query: str, entries: tuple[KnowledgeEntry, ...] = KB_ENTRIES) -> list[KnowledgeHit]:
    """
    Score every KB entry against ``query``.

    Scoring: an exact keyword-phrase hit is worth the phrase length (multi-word phrases
    therefore dominate single tokens); a title-token overlap adds 0.25 per token. Ties are
    resolved alphabetically by ``entry_id`` so retrieval is deterministic.
    """
    normalised = unicodedata.normalize("NFKC", query or "").lower()
    query_tokens = set(tokenize(query))
    hits: list[KnowledgeHit] = []

    for entry in entries:
        score = 0.0
        matched: list[str] = []
        for phrase in entry.keywords:
            if phrase in normalised:
                score += 1.0 + 0.5 * len(phrase.split())
                matched.append(phrase)
        title_tokens = set(tokenize(entry.title))
        overlap = query_tokens & title_tokens
        if overlap:
            score += 0.25 * len(overlap)
        if score > 0:
            hits.append(KnowledgeHit(entry=entry, score=round(score, 3), matched=tuple(matched)))

    hits.sort(key=lambda hit: (-hit.score, hit.entry.entry_id))
    return hits


def search(query: str, top_k: int = 3, min_score: float = 0.75) -> list[KnowledgeHit]:
    """Return the best ``top_k`` entries scoring at least ``min_score``."""
    top_k = max(1, int(top_k))
    return [hit for hit in score_entries(query) if hit.score >= min_score][:top_k]


FALLBACK_ANSWER: str = (
    "That is inside the machine-safety and maintenance domain, but I do not have a reviewed "
    "answer for the exact wording.\n\n"
    "Try naming the system (engine, hydraulics, undercarriage, fuel, emissions, safety "
    "interlock, proximity radar) plus what you observed - for example *\"hydraulic oil "
    "running hot while digging\"*. For torque values, pressures and fluid specifications, "
    "always use your machine's **Operation & Maintenance Manual (OMM)** or call the Cat "
    "dealer service desk. Anything that looks unsafe: **park, shut down, lock out / tag out**."
)


def answer(query: str, top_k: int = 3) -> tuple[str, tuple[str, ...]]:
    """
    Produce a deterministic answer plus the ids of the entries used.

    Returns ``(markdown_text, used_entry_ids)``. When nothing scores above the threshold the
    static :data:`FALLBACK_ANSWER` is returned, so the copilot never invents numbers.
    """
    hits = search(query, top_k=top_k)
    if not hits:
        return FALLBACK_ANSWER, ()

    blocks = [hit.entry.render() for hit in hits]
    used = tuple(hit.entry.entry_id for hit in hits)
    header = (
        "Here is the reviewed guidance from the **CAT-Pal knowledge base** "
        "(offline mode - answers are selected, not generated):"
    )
    footer = (
        "_Verify torque values, pressures and fluid specs against your machine's OMM, and "
        "raise a service request for anything that looks like a defect._"
    )
    return "\n\n".join([header] + blocks + [footer]), used


__all__ = [
    "STOPWORDS",
    "KnowledgeEntry",
    "KB_ENTRIES",
    "KB_BY_ID",
    "KnowledgeHit",
    "tokenize",
    "score_entries",
    "search",
    "answer",
    "FALLBACK_ANSWER",
]
