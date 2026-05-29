from weather_tmax_bot.data.dwd_station_metadata import parse_station_metadata_text, select_station_for_airport


def test_parse_and_select_eddm_station():
    text = """
Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge Stationsname Bundesland
01262 19920519 20260528            446     48.3477   11.8134 Muenchen-Flughafen Bayern Frei
03379 19970710 20260528            515     48.1632   11.5429 Muenchen-Stadt Bayern Frei
"""
    stations = parse_station_metadata_text(text)
    selected = select_station_for_airport(stations, 48.3538, 11.7861)
    assert selected.station_id == "01262"
    assert selected.date_from.isoformat() == "1992-05-19"
