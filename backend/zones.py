import os
from threading import Lock


LOW_RISK_MAX_PERCENT = float(os.getenv("LOW_RISK_MAX_PERCENT", "5"))
HIGH_RISK_MIN_PERCENT = float(os.getenv("HIGH_RISK_MIN_PERCENT", "20"))


def _new_zone(zone_id):
    return {
        "zone_id": zone_id,
        "zone": f"Zone {zone_id}",
        "plants_scanned": 0,
        "healthy": 0,
        "pest": 0,
        "disease": 0,
        "affected": 0,
        "affected_percentage": 0.0,
        "risk": "NOT_SCANNED",
        "action": "Scan this zone to assess crop health",
    }


class ZoneStore:
    def __init__(self, zone_count=6):
        self._zones = {str(index): _new_zone(index) for index in range(1, zone_count + 1)}
        self._lock = Lock()

    def _get(self, zone_id):
        zone = self._zones.get(str(zone_id))
        if zone is None:
            raise KeyError(zone_id)
        return zone

    def record(self, zone_id, predictions):
        with self._lock:
            zone = self._get(zone_id)
            for prediction in predictions:
                if prediction not in ("Healthy", "Pest", "Disease"):
                    continue
                zone["plants_scanned"] += 1
                zone[prediction.lower()] += 1
                if prediction in ("Pest", "Disease"):
                    zone["affected"] += 1
            self._refresh(zone)
            return dict(zone)

    def get(self, zone_id):
        with self._lock:
            return dict(self._get(zone_id))

    def all(self):
        with self._lock:
            return [dict(zone) for zone in self._zones.values()]

    def reset(self, zone_id):
        with self._lock:
            self._get(zone_id)
            self._zones[str(zone_id)] = _new_zone(int(zone_id))
            return dict(self._zones[str(zone_id)])

    @staticmethod
    def _refresh(zone):
        total = zone["plants_scanned"]
        zone["affected_percentage"] = round(zone["affected"] / total * 100, 2) if total else 0.0
        if total == 0:
            zone["risk"] = "NOT_SCANNED"
            zone["action"] = "Scan this zone to assess crop health"
        elif zone["affected_percentage"] >= HIGH_RISK_MIN_PERCENT:
            zone["risk"] = "HIGH"
            zone["action"] = "Priority inspection required"
        elif zone["affected_percentage"] > LOW_RISK_MAX_PERCENT:
            zone["risk"] = "MEDIUM"
            zone["action"] = "Monitor closely and inspect again"
        else:
            zone["risk"] = "LOW"
            zone["action"] = "No targeted treatment indicated by the current scan"


zones = ZoneStore()