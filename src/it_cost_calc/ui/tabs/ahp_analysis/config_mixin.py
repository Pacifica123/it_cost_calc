import tkinter as tk
from tkinter import messagebox


class AHPConfigurationMixin:
    def _add_config_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Новая конфигурация")
        tk.Label(popup, text="ID конфигурации").pack()
        id_entry = tk.Entry(popup)
        id_entry.pack()
        tk.Label(popup, text="People (число людей)").pack()
        people_entry = tk.Entry(popup)
        people_entry.pack()
        people_entry.insert(0, "1")

        def save():
            cid = id_entry.get().strip()
            if not cid:
                messagebox.showerror("Error", "ID не может быть пустым")
                return
            people = int(people_entry.get() or 0)
            self.configurations[cid] = {"devices": [], "meta": {"people": people}}
            self.cfg_table.insert("", "end", iid=cid, values=(cid, people, 0))
            popup.destroy()

        tk.Button(popup, text="Сохранить", command=save).pack()

    def _delete_selected_config(self):
        sel = self.cfg_table.selection()
        if not sel:
            return
        for iid in sel:
            if iid in self.configurations:
                del self.configurations[iid]
            self.cfg_table.delete(iid)
        self.dev_table.delete(*self.dev_table.get_children())

    def _on_select_config(self, event):
        sel = self.cfg_table.selection()
        if not sel:
            return
        iid = sel[0]
        cfg = self.configurations.get(iid, {"devices": [], "meta": {}})
        # populate dev_table
        self.dev_table.delete(*self.dev_table.get_children())
        for idx, d in enumerate(cfg["devices"]):
            vals = (
                d.get("role", ""),
                d.get("vendor", ""),
                d.get("cpu_score", 0),
                d.get("ram_score", 0),
                d.get("energy", 0),
                d.get("cost", 0),
                d.get("reliability", {}).get("low", ""),
                d.get("reliability", {}).get("high", ""),
            )
            self.dev_table.insert("", "end", iid=f"{iid}_dev_{idx}", values=vals)

    def _add_device_popup(self):
        sel = self.cfg_table.selection()
        if not sel:
            messagebox.showerror("Error", "Сначала выберите конфигурацию")
            return
        cfg_id = sel[0]
        popup = tk.Toplevel(self)
        popup.title("Добавить устройство")

        entries = {}
        for label, default in [
            ("role", "client"),
            ("vendor", ""),
            ("cpu_score", "0.0"),
            ("ram_score", "0.0"),
            ("energy", "0.0"),
            ("cost", "0.0"),
            ("rel_low", "0.5"),
            ("rel_high", "0.9"),
        ]:
            tk.Label(popup, text=label).pack(anchor="w")
            e = tk.Entry(popup)
            e.pack(fill="x")
            e.insert(0, default)
            entries[label] = e

        def save_dev():
            d = {
                "role": entries["role"].get(),
                "vendor": entries["vendor"].get(),
                "cpu_score": float(entries["cpu_score"].get() or 0.0),
                "ram_score": float(entries["ram_score"].get() or 0.0),
                "energy": float(entries["energy"].get() or 0.0),
                "cost": float(entries["cost"].get() or 0.0),
                "reliability": {
                    "low": float(entries["rel_low"].get() or 0.0),
                    "high": float(entries["rel_high"].get() or 0.0),
                },
            }
            self.configurations[cfg_id]["devices"].append(d)
            # update table counts
            cnt = len(self.configurations[cfg_id]["devices"])
            people = self.configurations[cfg_id]["meta"].get("people", 0)
            self.cfg_table.item(cfg_id, values=(cfg_id, people, cnt))
            self._on_select_config(None)
            popup.destroy()

        tk.Button(popup, text="Добавить", command=save_dev).pack()

    def _delete_device(self):
        sel = self.dev_table.selection()
        if not sel:
            return
        item = sel[0]
        # parse id and index
        parts = item.split("_dev_")
        if len(parts) != 2:
            return
        cfg_id, idx = parts[0], int(parts[1])
        if cfg_id in self.configurations and idx < len(self.configurations[cfg_id]["devices"]):
            del self.configurations[cfg_id]["devices"][idx]
            # rebuild dev table
            self._on_select_config(None)
            # update devices_count
            cnt = len(self.configurations[cfg_id]["devices"])
            people = self.configurations[cfg_id]["meta"].get("people", 0)
            self.cfg_table.item(cfg_id, values=(cfg_id, people, cnt))
