# ausgedachtes handbuch für tests + eval

FILENAME = "mitarbeiterhandbuch_nordlicht.pdf"

PAGES = [
    (
        "Mitarbeiterhandbuch der Nordlicht Software GmbH",
        "Willkommen bei der Nordlicht Software GmbH. Das Unternehmen wurde 2014 in Kiel "
        "gegründet und entwickelt Software für die Logistikbranche. Heute arbeiten 140 "
        "Menschen an den Standorten Kiel, Hamburg und Leipzig.\n\n"
        "Dieses Handbuch fasst die wichtigsten Regeln für den Arbeitsalltag zusammen. Bei "
        "Fragen hilft das People-Team unter people@nordlicht.example.",
    ),
    (
        "Arbeitszeit und Homeoffice",
        "Die regelmäßige Wochenarbeitszeit beträgt 38 Stunden. Es gilt Gleitzeit mit einer "
        "Kernarbeitszeit von 10 bis 15 Uhr. Überstunden werden auf einem Zeitkonto erfasst "
        "und können als Freizeit ausgeglichen werden; das Konto darf höchstens 60 Plusstunden "
        "aufweisen.\n\n"
        "Mobiles Arbeiten ist an bis zu drei Tagen pro Woche möglich. Für die Ausstattung "
        "des Arbeitsplatzes zu Hause gibt es einmalig ein Budget von 600 Euro.",
    ),
    (
        "Urlaub und Krankheit",
        "Alle Vollzeitkräfte erhalten 30 Urlaubstage pro Kalenderjahr. Der 24. und der 31. "
        "Dezember sind zusätzlich arbeitsfrei. Resturlaub muss bis zum 31. März des "
        "Folgejahres genommen werden, sonst verfällt er.\n\n"
        "Im Krankheitsfall meldest du dich bis 9 Uhr bei deiner Teamleitung. Ab dem dritten "
        "Krankheitstag ist eine ärztliche Arbeitsunfähigkeitsbescheinigung nötig.",
    ),
    (
        "Weiterbildung",
        "Jede Mitarbeiterin und jeder Mitarbeiter hat ein Weiterbildungsbudget von 1.500 Euro "
        "pro Jahr sowie fünf zusätzliche Bildungstage. Das Budget kann für Kurse, "
        "Konferenzen, Bücher und Zertifizierungen genutzt werden.\n\n"
        "Anträge werden über das interne Tool 'Lernpfad' gestellt und von der Teamleitung "
        "innerhalb von fünf Arbeitstagen freigegeben.",
    ),
    (
        "Reisekosten",
        "Dienstreisen mit der Bahn erfolgen in der 2. Klasse; ab einer Fahrtzeit von vier "
        "Stunden ist die 1. Klasse erlaubt. Für Übernachtungen gilt eine Obergrenze von "
        "140 Euro pro Nacht, in München und Frankfurt 180 Euro.\n\n"
        "Reisekostenabrechnungen müssen spätestens drei Monate nach der Reise mit allen "
        "Belegen eingereicht werden. Die Kilometerpauschale für das eigene Auto beträgt "
        "0,30 Euro pro Kilometer.",
    ),
    (
        "IT-Sicherheit",
        "Passwörter müssen mindestens 14 Zeichen lang sein. Für alle Firmenkonten ist die "
        "Zwei-Faktor-Authentifizierung Pflicht. Verdächtige E-Mails leitest du an "
        "security@nordlicht.example weiter und öffnest keine Anhänge.\n\n"
        "Firmenlaptops werden automatisch gesperrt, wenn sie fünf Minuten nicht benutzt "
        "werden. Der Verlust eines Geräts muss innerhalb von einer Stunde gemeldet werden.",
    ),
]


def write_pdf(path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_margins(20, 20)
    for title, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 9, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=11)
        for para in body.split("\n\n"):
            pdf.multi_cell(0, 6, para)
            pdf.ln(3)
    pdf.output(str(path))
