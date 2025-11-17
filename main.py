"""
Cisco Spaces Occupancy + Context Demo
-------------------------------------
Synthetic occupancy data generator, RDF knowledge graph builder, and
Streamlit UI that surfaces graph-aware, LLM-backed explanations.

Run the UI with:
    streamlit run main.py

Required packages:
    pandas, numpy, rdflib, streamlit, openai (optional for live LLM calls)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit not always installed for tests
    st = None  # type: ignore

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


ARTIFACT_DIR = Path("artifacts")
WEEK_START = datetime(2025, 6, 9, 6, 0, 0)
WEEK_DAYS = 7
HOURS = range(7, 20)  # 7:00 through 19:00 inclusive


@dataclass
class FloorMetadata:
    floor_id: str
    number: int
    label: str
    purpose: str
    reserved_for: str
    status: str
    notes: str
    restrictions: str
    base_occupancy: int
    preferred_peak_hour: int


@dataclass
class ContextGap:
    reason: str
    suggested_question: str
    peer_hint: str


class ContextVault:
    """Persistent store for user-supplied context enrichments."""

    def __init__(self, path: Path = ARTIFACT_DIR / "context_overrides.json") -> None:
        self.path = path
        self.entries: List[Dict[str, str]] = self._load()

    def _load(self) -> List[Dict[str, str]]:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                print(f"[ContextVault] Failed to read overrides ({exc}); starting fresh.")
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self.entries, fp, indent=2)

    def add_entry(
        self,
        floor_id: str,
        predicate: str,
        value: str,
        source_note: str = "user_follow_up",
    ) -> Dict[str, str]:
        entry = {
            "floor_id": floor_id,
            "predicate": predicate,
            "value": value,
            "source_note": source_note,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.entries.append(entry)
        self._save()
        return entry

    def facts_for_floor(self, floor_id: str) -> List[Dict[str, str]]:
        return [entry for entry in self.entries if entry["floor_id"] == floor_id]

    def apply_to_graph(self, graph: Graph, ns: Namespace, floor_nodes: Dict[str, URIRef]) -> None:
        for entry in self.entries:
            subj = floor_nodes.get(entry["floor_id"])
            if subj is None:
                continue
            predicate = ns[entry["predicate"]]
            graph.add((subj, predicate, Literal(entry["value"])))

    def as_list(self) -> List[Dict[str, str]]:
        return list(self.entries)


def create_floor_metadata() -> List[FloorMetadata]:
    """Define the structured schema for each floor."""
    return [
        FloorMetadata(
            floor_id="Floor1",
            number=1,
            label="Lobby & Visitor Pavilion",
            purpose="Reception, visitor management, cafe, experience center",
            reserved_for="AllEmployeesAndGuests",
            status="Open",
            notes="Hosts daily executive briefings and high-volume cafe traffic.",
            restrictions="Visitor badge required past turnstiles.",
            base_occupancy=85,
            preferred_peak_hour=11,
        ),
        FloorMetadata(
            floor_id="Floor2",
            number=2,
            label="Collaboration Hub (Under Renovation)",
            purpose="Agile project rooms and large briefing studio",
            reserved_for="FacilitiesTeamOnly",
            status="Renovation (Jun 1–15)",
            notes="Construction crews modernizing briefing studio; sensors stay online for safety compliance.",
            restrictions="Hard-hat access, max 10 people per hour.",
            base_occupancy=60,
            preferred_peak_hour=10,
        ),
        FloorMetadata(
            floor_id="Floor3",
            number=3,
            label="Innovation & Engineering Pods",
            purpose="Software engineering, hack spaces, UX labs",
            reserved_for="ProductEngineering",
            status="Open",
            notes="Hosts rotating hackathons; busiest Tues–Thu afternoons.",
            restrictions="Badge + NDA for lab wing.",
            base_occupancy=130,
            preferred_peak_hour=14,
        ),
        FloorMetadata(
            floor_id="Floor4",
            number=4,
            label="Customer Experience Center",
            purpose="Partner enablement, demo theaters, training",
            reserved_for="CXEnablement",
            status="Open",
            notes="Peak demand around on-site trainings; steady but moderate.",
            restrictions="Escort visitors outside of demo zones.",
            base_occupancy=95,
            preferred_peak_hour=13,
        ),
        FloorMetadata(
            floor_id="Floor5",
            number=5,
            label="Executive Offices (CXOs only)",
            purpose="Executive offices, boardroom, strategy war-room",
            reserved_for="ExecutiveLeadership",
            status="Restricted",
            notes="Badge restricted; typically empty on weekends and evenings.",
            restrictions="24/7 security post, invite-only access.",
            base_occupancy=25,
            preferred_peak_hour=12,
        ),
    ]


def create_unstructured_context(floors: Sequence[FloorMetadata]) -> List[str]:
    """Simulate memos or news articles that mention floor facts."""
    return [
        (
            "Facilities bulletin (May 28): Floor 2 will remain closed for fit-out work "
            "between June 1–15. Only construction contractors and the facilities team "
            "are permitted, so expect near-zero badge events despite sensors staying online "
            "for safety monitoring."
        ),
        (
            "Security advisory: The executive suite on Floor 5 is CXO-only. "
            "Weekend building patrols lock the elevator destination controls, "
            "so low or zero occupancy readings on Saturdays and Sundays are normal."
        ),
        (
            "Employee newsletter: Engineering teams on Floor 3 scheduled a hackathon "
            "on June 12–13, meaning afternoon occupancy spikes are expected in the innovation pods."
        ),
    ]


def floor_schema_to_json(floors: Sequence[FloorMetadata]) -> Dict[str, object]:
    """Represent the structured metadata as a JSON schema-like document."""
    return {
        "building": {
            "name": "Cisco Spaces HQ (Demo Tower)",
            "address": "170 W Tasman Dr, San Jose, CA",
        },
        "floors": [
            {
                "floorId": f.floor_id,
                "number": f.number,
                "label": f.label,
                "purpose": f.purpose,
                "reservedFor": f.reserved_for,
                "status": f.status,
                "notes": f.notes,
                "restrictions": f.restrictions,
            }
            for f in floors
        ],
    }


def _hourly_peak(hour: int, peak: int) -> float:
    """Return a smooth curve with a peak around the preferred hour."""
    return float(np.exp(-0.5 * ((hour - peak) / 2.5) ** 2))


def generate_occupancy_dataframe(
    floors: Sequence[FloorMetadata], seed: int = 11
) -> pd.DataFrame:
    """Create timestamped occupancy readings for each floor over a week."""
    rng = np.random.default_rng(seed)
    records: List[Dict[str, object]] = []
    days = [WEEK_START + timedelta(days=i) for i in range(WEEK_DAYS)]

    for floor in floors:
        for day in days:
            is_weekend = day.weekday() >= 5
            for hour in HOURS:
                timestamp = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                hour_factor = _hourly_peak(hour, floor.preferred_peak_hour) + 0.25
                day_factor = 0.55 if is_weekend else 1.0
                base_count = floor.base_occupancy * hour_factor * day_factor

                if floor.number == 2 and datetime(2025, 6, 1) <= timestamp <= datetime(
                    2025, 6, 15, 23, 59
                ):
                    base_count = rng.normal(loc=1.5, scale=0.6)
                elif floor.number == 5:
                    exec_factor = 0.35 if not is_weekend else 0.05
                    base_count = max(0.1, floor.base_occupancy * hour_factor * exec_factor)
                elif floor.number == 3 and day.weekday() in (2, 3):  # Wed/Thu hackathon
                    base_count *= 1.15

                noise = rng.normal(loc=0, scale=max(3, floor.base_occupancy * 0.05))
                count = max(0, int(round(base_count + noise)))

                records.append(
                    {
                        "timestamp": timestamp,
                        "floor_id": floor.floor_id,
                        "floor_number": floor.number,
                        "count": count,
                        "sensor_id": f"S-{floor.number:02d}-Z1",
                        "weekday": timestamp.strftime("%A"),
                    }
                )

    df = pd.DataFrame.from_records(records)
    df["date"] = df["timestamp"].dt.date
    return df


def persist_artifacts(
    occupancy_df: pd.DataFrame, floor_schema: Dict[str, object], memos: Sequence[str]
) -> Dict[str, Path]:
    """Save synthetic datasets so the user can inspect them directly."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / "occupancy_week.csv"
    json_path = ARTIFACT_DIR / "floor_schema.json"
    memos_path = ARTIFACT_DIR / "context_memos.txt"

    occupancy_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(floor_schema, fp, indent=2)
    with open(memos_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n---\n\n".join(memos))

    return {"occupancy_csv": csv_path, "floor_schema_json": json_path, "memos_txt": memos_path}


class TripletExtractor:
    """Extract subject-predicate-object triplets from unstructured memos."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_ready = bool(self.api_key and openai)
        if self.llm_ready and openai:
            openai.api_key = self.api_key

    def extract(self, memos: Sequence[str]) -> List[Dict[str, str]]:
        if self.llm_ready:
            try:
                return self._extract_with_llm(memos)
            except Exception as exc:  # pragma: no cover - depends on external API
                print(f"[TripletExtractor] Falling back to heuristics: {exc}")
        return self._heuristic_extract(memos)

    def _extract_with_llm(self, memos: Sequence[str]) -> List[Dict[str, str]]:
        assert openai is not None  # for mypy
        joined = "\n\n".join(memos)
        prompt = (
            "Extract building knowledge triplets from the memos below. "
            "Return strictly valid JSON in the following format:\n"
            '[{"subject":"Floor2","predicate":"status","object":"Under Renovation"}, ...]\n'
            "Use short predicate names (status, reservedFor, event, restriction).\n\n"
            f"Memos:\n{joined}"
        )
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": "You convert text into RDF-style triplets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [
                    {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]}
                    for t in parsed
                    if {"subject", "predicate", "object"} <= t.keys()
                ]
        except Exception:
            pass
        return self._heuristic_extract(memos)

    @staticmethod
    def _heuristic_extract(memos: Sequence[str]) -> List[Dict[str, str]]:
        triplets: List[Dict[str, str]] = []
        for memo in memos:
            lower = memo.lower()
            floor_matches = re.findall(r"floor\s*(\d)", memo, re.IGNORECASE)
            for num_str in floor_matches:
                floor_id = f"Floor{num_str}"
                if "renovation" in lower or "closed" in lower:
                    triplets.append(
                        {"subject": floor_id, "predicate": "status", "object": "Under Renovation"}
                    )
                if "executive" in lower or "cxo" in lower:
                    triplets.append(
                        {"subject": floor_id, "predicate": "reservedFor", "object": "ExecutiveLeadership"}
                    )
                if "hackathon" in lower:
                    triplets.append(
                        {
                            "subject": floor_id,
                            "predicate": "event",
                            "object": "Hackathon June 12-13",
                        }
                    )
                if "contractor" in lower or "facilities" in lower:
                    triplets.append(
                        {
                            "subject": floor_id,
                            "predicate": "restriction",
                            "object": "Contractors and facilities team only",
                        }
                    )
        return triplets


class KnowledgeGraphBuilder:
    """Construct an RDF graph combining structured, unstructured, and sensor data."""

    def __init__(
        self,
        floors: Sequence[FloorMetadata],
        occupancy_df: pd.DataFrame,
        memos: Sequence[str],
        triplets: Sequence[Dict[str, str]],
        floor_schema: Dict[str, object],
        context_vault: Optional[ContextVault] = None,
    ) -> None:
        self.floors = floors
        self.occupancy_df = occupancy_df
        self.memos = memos
        self.triplets = triplets
        self.floor_schema = floor_schema
        self.graph = Graph()
        self.ns = Namespace("https://cisco-spaces.demo/graph#")
        self.floor_nodes: Dict[str, URIRef] = {}
        self.context_vault = context_vault

    def build(self) -> Graph:
        self.graph.bind("spaces", self.ns)
        self._add_floors()
        self._add_schema_metadata()
        self._add_unstructured_triplets()
        self._add_memos()
        self._link_sensor_records()
        self._include_context_vault()
        return self.graph

    def _add_floors(self) -> None:
        for floor in self.floors:
            node = self.ns[floor.floor_id]
            self.floor_nodes[floor.floor_id] = node
            self.graph.add((node, RDF.type, self.ns.Floor))
            self.graph.add((node, RDFS.label, Literal(floor.label)))
            self.graph.add((node, self.ns.floorNumber, Literal(floor.number, datatype=XSD.integer)))
            self.graph.add((node, self.ns.purpose, Literal(floor.purpose)))
            self.graph.add((node, self.ns.status, Literal(floor.status)))
            self.graph.add((node, self.ns.reservedFor, Literal(floor.reserved_for)))
            self.graph.add((node, self.ns.restrictions, Literal(floor.restrictions)))
            self.graph.add((node, self.ns.notes, Literal(floor.notes)))

    def _add_schema_metadata(self) -> None:
        building = URIRef(self.ns["BuildingHQ"])
        self.graph.add((building, RDF.type, self.ns.Building))
        self.graph.add((building, RDFS.label, Literal(self.floor_schema["building"]["name"])))
        for floor in self.floors:
            self.graph.add((building, self.ns.hasFloor, self.floor_nodes[floor.floor_id]))

    def _add_unstructured_triplets(self) -> None:
        for idx, triplet in enumerate(self.triplets):
            subject = self.floor_nodes.get(triplet["subject"], self.ns[triplet["subject"]])
            predicate = self.ns[triplet["predicate"]]
            obj_value = triplet["object"]
            obj_node = (
                self.floor_nodes.get(obj_value)
                if obj_value in self.floor_nodes
                else Literal(obj_value)
            )
            self.graph.add((subject, predicate, obj_node))
            evidence = URIRef(self.ns[f"tripletEvidence/{idx}"])
            self.graph.add((evidence, RDF.type, self.ns.Evidence))
            self.graph.add((evidence, self.ns.supports, subject))
            self.graph.add((evidence, self.ns.statement, Literal(json.dumps(triplet))))

    def _add_memos(self) -> None:
        for idx, memo in enumerate(self.memos):
            memo_node = URIRef(self.ns[f"memo/{idx}"])
            self.graph.add((memo_node, RDF.type, self.ns.Memo))
            self.graph.add((memo_node, self.ns.text, Literal(memo)))
            mentioned_floors = re.findall(r"Floor\s*(\d)", memo, flags=re.IGNORECASE)
            for num in mentioned_floors:
                floor_id = f"Floor{num}"
                if floor_id in self.floor_nodes:
                    self.graph.add((memo_node, self.ns.mentions, self.floor_nodes[floor_id]))

    def _link_sensor_records(self) -> None:
        for idx, row in self.occupancy_df.iterrows():
            floor_node = self.floor_nodes[row["floor_id"]]
            reading_id = f"reading/{row['floor_id']}/{row['timestamp'].isoformat()}"
            reading_node = URIRef(self.ns[reading_id])
            self.graph.add((reading_node, RDF.type, self.ns.SensorRecord))
            self.graph.add(
                (reading_node, self.ns.timestamp, Literal(row["timestamp"].isoformat(), datatype=XSD.dateTime))
            )
            self.graph.add(
                (reading_node, self.ns.occupancyCount, Literal(int(row["count"]), datatype=XSD.integer))
            )
            self.graph.add((reading_node, self.ns.sensorId, Literal(row["sensor_id"])))
            self.graph.add((reading_node, self.ns.observedOn, floor_node))

    def _include_context_vault(self) -> None:
        if not self.context_vault:
            return
        self.context_vault.apply_to_graph(self.graph, self.ns, self.floor_nodes)

    def stats(self) -> Dict[str, int]:
        return {
            "triples": len(self.graph),
            "floors": len(self.floors),
            "sensor_records": len(self.occupancy_df),
            "memos": len(self.memos),
        }


class GraphQueryEngine:
    """Helper to retrieve facts relevant to a natural-language question."""

    FLOOR_PATTERN = re.compile(r"floor\s*(\d)", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"(June|Jun)\s+(\d{1,2})", re.IGNORECASE)
    MONTHS = {"june": 6, "jun": 6}
    PEER_HINTS = {
        "floor_profile": (
            "Peer insight: Workplace Ops teams log a short floor profile (purpose, badge policy, critical rooms)."
        ),
        "date_event": (
            "Peer insight: Leading customers track blackout dates like renovations or summits next to occupancy exports."
        ),
        "policy_note": (
            "Peer insight: Clients attach memo-style policy notes (e.g., 'CXO-only after 6pm') to each floor entity."
        ),
    }

    def __init__(
        self,
        graph: Graph,
        namespace: Namespace,
        occupancy_df: pd.DataFrame,
        floors: Sequence[FloorMetadata],
        memos: Sequence[str],
        artifacts: Dict[str, Path],
        context_vault: Optional[ContextVault] = None,
    ) -> None:
        self.graph = graph
        self.ns = namespace
        self.occupancy_df = occupancy_df
        self.floors = {f.floor_id: f for f in floors}
        self.memos = list(memos)
        self.artifacts = artifacts
        self.context_vault = context_vault
        self.memo_index: Dict[str, List[Tuple[int, str]]] = {}
        for idx, memo in enumerate(self.memos):
            mentioned = re.findall(r"floor\s*(\d)", memo, re.IGNORECASE)
            for num in mentioned:
                floor_id = f"Floor{num}"
                self.memo_index.setdefault(floor_id, []).append((idx, memo.strip()))

    def retrieve_context(
        self, question: str
    ) -> Tuple[str, List[str], Optional[datetime], List[ContextGap]]:
        requested_floors = self._detect_floors(question)
        missing_floors = [f for f in requested_floors if f not in self.floors]
        floors = [f for f in requested_floors if f in self.floors]
        date = self._detect_date(question)
        if not floors:
            floors = list(self.floors.keys())

        snippets: List[str] = []
        for floor_id in floors:
            snippets.append(self._format_floor_context(floor_id, date))

        gaps: List[ContextGap] = []
        gaps.extend(self._floor_gaps(missing_floors))
        gaps.extend(self._date_gaps(floors, date))
        gaps.extend(self._policy_gaps(requested_floors or floors))

        return "\n\n".join(snippets), floors, date, gaps

    def _detect_floors(self, question: str) -> List[str]:
        matches = self.FLOOR_PATTERN.findall(question)
        return [f"Floor{m}" for m in matches]

    def _detect_date(self, question: str) -> Optional[datetime]:
        match = self.DATE_PATTERN.search(question)
        if not match:
            return None
        month = self.MONTHS.get(match.group(1).lower())
        if not month:
            return None
        day = int(match.group(2))
        return datetime(2025, month, day)

    def _format_floor_context(self, floor_id: str, target_date: Optional[datetime]) -> str:
        node = self.ns[floor_id]
        facts = {
            "label": self._first_literal(node, RDFS.label),
            "purpose": self._first_literal(node, self.ns.purpose),
            "status": self._first_literal(node, self.ns.status),
            "reserved_for": self._first_literal(node, self.ns.reservedFor),
            "restrictions": self._first_literal(node, self.ns.restrictions),
        }
        df = self.occupancy_df[self.occupancy_df["floor_id"] == floor_id]
        summary = df["count"].agg(["mean", "max", "min"]).to_dict()
        context_lines = [
            f"{floor_id} ({facts['label']}): purpose={facts['purpose']} "
            f"[source: {self._source_link('floor_schema_json')}]",
            f"Status={facts['status']} | Reserved for {facts['reserved_for']} | Restrictions={facts['restrictions']} "
            f"[source: {self._source_link('floor_schema_json')}]",
            f"Weekly occupancy stats: avg={summary['mean']:.1f}, max={summary['max']}, min={summary['min']} "
            f"[source: {self._source_link('occupancy_csv')}]",
        ]
        if target_date is not None:
            date_df = df[df["date"] == target_date.date()]
            if not date_df.empty:
                context_lines.append(
                    f"On {target_date.strftime('%Y-%m-%d')} occupancy ranged "
                    f"{date_df['count'].min()}–{date_df['count'].max()} (median {int(date_df['count'].median())}) "
                    f"[source: {self._source_link('occupancy_csv')}]"
                )
            else:
                context_lines.append(
                    f"No sensor data for {floor_id} on {target_date.strftime('%Y-%m-%d')}."
                )
        context_lines.extend(self._memo_snippets(floor_id))
        context_lines.extend(self._vault_snippets(floor_id))
        return "\n".join(context_lines)

    def _first_literal(self, subject: URIRef, predicate: URIRef) -> str:
        for _, _, obj in self.graph.triples((subject, predicate, None)):
            return str(obj)
        return "Unknown"

    def _memo_snippets(self, floor_id: str) -> List[str]:
        snippets: List[str] = []
        for idx, memo in self.memo_index.get(floor_id, []):
            excerpt = memo if len(memo) <= 220 else memo[:217] + "..."
            snippets.append(
                f"Memo evidence #{idx + 1}: {excerpt} "
                f"[source: {self._source_link('memos_txt')}#memo-{idx + 1}]"
            )
        return snippets

    def _source_link(self, key: str) -> str:
        path = self.artifacts.get(key)
        return path.as_posix() if path else key

    def _vault_snippets(self, floor_id: str) -> List[str]:
        if not self.context_vault:
            return []
        snippets: List[str] = []
        for idx, entry in enumerate(self.context_vault.facts_for_floor(floor_id), start=1):
            snippets.append(
                f"Context vault fact #{idx}: {entry['predicate']}={entry['value']} "
                f"(added {entry['timestamp']}, note={entry.get('source_note', 'user')}) "
                f"[source: {self._source_link('context_overrides_json')}]"
            )
        return snippets

    def _floor_gaps(self, missing_floors: Sequence[str]) -> List[ContextGap]:
        gaps: List[ContextGap] = []
        for floor_id in missing_floors:
            gaps.append(
                ContextGap(
                    reason=f"{floor_id} is not modeled in the current building graph.",
                    suggested_question=(
                        f"Can you summarize {floor_id}'s purpose, access policy, and monitoring status?"
                    ),
                    peer_hint=self.PEER_HINTS["floor_profile"],
                )
            )
        return gaps

    def _date_gaps(self, floors: Sequence[str], target_date: Optional[datetime]) -> List[ContextGap]:
        gaps: List[ContextGap] = []
        if not target_date:
            return gaps
        for floor_id in floors:
            date_df = self.occupancy_df[
                (self.occupancy_df["floor_id"] == floor_id)
                & (self.occupancy_df["date"] == target_date.date())
            ]
            if date_df.empty:
                gaps.append(
                    ContextGap(
                        reason=(
                            f"No sensor coverage for {floor_id} on {target_date.strftime('%Y-%m-%d')}."
                        ),
                        suggested_question=(
                            f"Was {floor_id} under maintenance, blocked badges, or a special event on "
                            f"{target_date.strftime('%b %d')}?"
                        ),
                        peer_hint=self.PEER_HINTS["date_event"],
                    )
                )
        return gaps

    def _policy_gaps(self, floors: Sequence[str]) -> List[ContextGap]:
        gaps: List[ContextGap] = []
        for floor_id in floors:
            has_memo = bool(self.memo_index.get(floor_id))
            has_vault = bool(self.context_vault and self.context_vault.facts_for_floor(floor_id))
            if not has_memo and not has_vault:
                gaps.append(
                    ContextGap(
                        reason=f"{floor_id} lacks memo evidence or manual overrides.",
                        suggested_question=(
                            f"Could you share a short note about {floor_id}'s operating norms or current projects?"
                        ),
                        peer_hint=self.PEER_HINTS["policy_note"],
                    )
                )
        return gaps


class LLMExplainer:
    """Answer natural language questions using graph retrieval + optional LLM."""

    def __init__(self, query_engine: GraphQueryEngine, model: str = "gpt-4o-mini"):
        self.query_engine = query_engine
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_ready = bool(self.api_key and openai)
        if self.llm_ready and openai:
            openai.api_key = self.api_key

    def answer(self, question: str) -> Tuple[str, str]:
        context, floors, date, gaps = self.query_engine.retrieve_context(question)
        if gaps:
            return self._gap_response(question, gaps), context
        if self.llm_ready:
            try:
                return self._answer_with_llm(question, context), context
            except Exception as exc:  # pragma: no cover - depends on external API
                fallback = self._fallback_answer(question, context, error=str(exc))
                return fallback, context
        fallback = self._fallback_answer(question, context)
        return fallback, context

    def _answer_with_llm(self, question: str, context: str) -> str:
        assert openai is not None  # for mypy
        completion = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a workplace operations analyst. Use only the provided facts "
                        "to explain occupancy anomalies. Reference floor numbers explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nFacts:\n{context}",
                },
            ],
            temperature=0.2,
        )
        return completion["choices"][0]["message"]["content"].strip()

    def _fallback_answer(self, question: str, context: str, error: Optional[str] = None) -> str:
        explanation = [
            "Offline reasoning (LLM unavailable):",
            f"- Question: {question}",
            "- Based on retrieved facts:",
        ]
        for line in context.splitlines():
            explanation.append(f"  • {line}")
        if error:
            explanation.append(f"[LLM error: {error}]")
        return "\n".join(explanation)

    def _gap_response(self, question: str, gaps: Sequence[ContextGap]) -> str:
        lines = [
            "Additional context required before I can explain confidently:",
            f"- Original question: {question}",
        ]
        for idx, gap in enumerate(gaps, start=1):
            lines.append(
                f"{idx}. Reason: {gap.reason}\n   Ask the user: {gap.suggested_question}\n   {gap.peer_hint}"
            )
        lines.append(
            "Once answered, call `record_context_entry(floor_id, predicate, value, source_note)` "
            "and re-run the analysis so the new fact persists in `artifacts/context_overrides.json`."
        )
        return "\n".join(lines)


def _build_data_bundle() -> Dict[str, object]:
    floors = create_floor_metadata()
    memos = create_unstructured_context(floors)
    floor_schema = floor_schema_to_json(floors)
    occupancy_df = generate_occupancy_dataframe(floors)
    context_vault = ContextVault()
    artifact_paths = persist_artifacts(occupancy_df, floor_schema, memos)
    artifact_paths["context_overrides_json"] = context_vault.path

    extractor = TripletExtractor()
    triplets = extractor.extract(memos)

    builder = KnowledgeGraphBuilder(
        floors, occupancy_df, memos, triplets, floor_schema, context_vault=context_vault
    )
    graph = builder.build()
    query_engine = GraphQueryEngine(
        graph,
        builder.ns,
        occupancy_df,
        floors,
        memos,
        artifact_paths,
        context_vault=context_vault,
    )
    explainer = LLMExplainer(query_engine)

    return {
        "floors": floors,
        "memos": memos,
        "floor_schema": floor_schema,
        "occupancy_df": occupancy_df,
        "triplets": triplets,
        "graph": graph,
        "graph_stats": builder.stats(),
        "artifacts": artifact_paths,
        "explainer": explainer,
        "context_vault": context_vault,
    }


def record_context_entry(
    floor_id: str,
    predicate: str,
    value: str,
    source_note: str = "user_follow_up",
) -> Dict[str, str]:
    """
    Persist a manual context fact that will be merged into the knowledge graph on the next run.

    Example:
        record_context_entry(
            "Floor4",
            "eventNote",
            "Partner innovation summit July 1 (floor closed to staff)",
            "Facilities reply 2025-11-17",
        )
    """
    vault = ContextVault()
    entry = vault.add_entry(floor_id, predicate, value, source_note)
    return entry


def is_streamlit_runtime() -> bool:
    return bool(st and getattr(st, "_is_running_with_streamlit", False))


def prepare_demo_data() -> Dict[str, object]:
    if st and is_streamlit_runtime():
        if "data_bundle" not in st.session_state:
            st.session_state["data_bundle"] = _build_data_bundle()
        return st.session_state["data_bundle"]
    return _build_data_bundle()


def render_streamlit_app(bundle: Dict[str, object]) -> None:
    if not st:
        raise RuntimeError("Streamlit is not installed. Run `pip install streamlit` first.")

    st.set_page_config(page_title="Cisco Spaces + Context Demo", layout="wide")
    st.title("Cisco Spaces Occupancy Demo with Context-Aware Reasoning")
    st.write(
        "This dashboard fabricates a week's worth of multi-floor occupancy, enriches it "
        "with floor policies and memos, and uses a knowledge graph to answer questions."
    )

    artifacts: Dict[str, Path] = bundle["artifacts"]  # type: ignore[assignment]
    st.sidebar.header("Saved artifacts")
    for label, path in artifacts.items():
        st.sidebar.write(f"{label}: `{path}`")
    st.sidebar.info(
        "Set the OPENAI_API_KEY env variable before launching Streamlit to enable live LLM responses."
    )

    occupancy_df: pd.DataFrame = bundle["occupancy_df"]  # type: ignore[assignment]
    st.subheader("Occupancy overview")
    st.caption("Synthetic sensor readings aggregated per hour for Floors 1–5, week of Jun 9, 2025.")
    st.dataframe(occupancy_df.head(20))

    pivot = occupancy_df.pivot_table(
        index="timestamp", columns="floor_id", values="count", aggfunc="mean"
    )
    st.line_chart(pivot)

    st.subheader("Floor schema (structured data)")
    st.json(bundle["floor_schema"])  # type: ignore[arg-type]

    st.subheader("Unstructured memos / news excerpts")
    for memo in bundle["memos"]:  # type: ignore[assignment]
        st.info(memo)

    st.subheader("Graph-derived facts")
    st.json(
        {
            "triplets_from_text": bundle["triplets"],  # type: ignore[arg-type]
            "graph_stats": bundle["graph_stats"],  # type: ignore[arg-type]
        }
    )

    st.subheader("Manual context overrides")
    vault_entries = bundle["context_vault"].as_list()  # type: ignore[assignment]
    if vault_entries:
        st.json(vault_entries)
    else:
        st.caption(
            "No manual overrides captured yet. Call `record_context_entry(...)` after gathering a clarification."
        )

    st.subheader("Ask the context-aware assistant")
    question = st.text_input(
        "Type a question about occupancy anomalies",
        value="Why is Floor 5 almost empty on Monday?",
    )
    if question:
        answer, supporting_facts = bundle["explainer"].answer(question)  # type: ignore[assignment]
        st.markdown("**Answer**")
        st.write(answer)
        with st.expander("Retrieved fact sheet"):
            st.text(supporting_facts)

    st.caption(
        "Demo data is synthetic. Knowledge graph built with RDFLib; sensor readings linked to floor entities."
    )


def run_cli_summary(bundle: Dict[str, object]) -> None:
    print("Streamlit UI not detected. Here is a textual summary instead:\n")
    print(f"- Synthetic occupancy rows: {len(bundle['occupancy_df'])}")
    print(f"- Knowledge graph triples: {bundle['graph_stats']['triples']}")
    print(f"- Triplets extracted from text: {len(bundle['triplets'])}")
    vault_entries = bundle["context_vault"].as_list()  # type: ignore[assignment]
    print(f"- Manual context overrides stored: {len(vault_entries)}")
    print("- Saved artifacts:")
    for label, path in bundle["artifacts"].items():  # type: ignore[assignment]
        print(f"  • {label}: {path}")
    print("\nRun `streamlit run main.py` to explore the interactive demo.")


def main() -> None:
    bundle = prepare_demo_data()
    if is_streamlit_runtime():
        render_streamlit_app(bundle)
    else:
        run_cli_summary(bundle)


if __name__ == "__main__":
    main()
