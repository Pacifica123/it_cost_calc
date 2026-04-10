from it_cost_calc.application.services.npv_report_service import NPVReportService


def test_npv_report_service_builds_rows_and_accumulated_points():
    service = NPVReportService()

    report = service.build_report(1000, 0.1, [500, 600])

    assert len(report["rows"]) == 3
    assert len(report["accumulated_points"]) == 3
    assert report["rows"][0]["year"] == 0
    assert isinstance(report["npv"], float)
