class CriteriaImportanceDataMixin:
    def _criteria(self):
        return self.case_data.get("criteria", []) if self.case_data else []

    def _alternatives(self):
        return self.case_data.get("alternatives", []) if self.case_data else []

    def _criterion_ids(self):
        return [c["id"] for c in self._criteria()]

    def _alternative_ids(self):
        return [a["id"] for a in self._alternatives()]

    def _criterion_name(self, criterion_id):
        for criterion in self._criteria():
            if criterion["id"] == criterion_id:
                return criterion["name"]
        return criterion_id

    def _alternative_name(self, alt_id):
        for alt in self._alternatives():
            if alt["id"] == alt_id:
                return alt["name"]
        return alt_id

    def _suggest_new_criterion_id(self):
        ids = set(self._criterion_ids())
        idx = 1
        while True:
            candidate = f"K{idx}"
            if candidate not in ids:
                return candidate
            idx += 1

    def _suggest_new_alternative_id(self):
        ids = set(self._alternative_ids())
        idx = 1
        while True:
            candidate = f"system{idx}"
            if candidate not in ids:
                return candidate
            idx += 1
