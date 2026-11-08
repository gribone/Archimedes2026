"""Cluster export: each profile with its description and the ΑΦΜ it contains.

For every profile (cluster) it lists the recommended new businesses (prospects, by ΑΦΜ, ranked) and,
for context, the existing customers in it. A prospect is "within the profile" when its venue lies
inside the profile's acceptance radius (the same rule as assignments.csv); otherwise it is an
exploratory suggestion whose nearest profile is this one.

Files: clusters.json and clusters.csv (machine-readable) and clusters.xlsx (one sheet per profile).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CSV_COLUMNS = ['profile_id', 'title', 'role', 'vat', 'company_id', 'company_name', 'city', 'municipality',
               'rank', 'profile_rank', 'score', 'within_profile', 'rationale']


def build_clusters(companies, profiles, fit, assignments, lookalikes):
    """Long table: one row per (profile, ΑΦΜ), role 'prospect' or 'customer'."""
    info = companies.set_index('company_id')[['vat', 'company_name_el', 'city', 'municipality_name']]
    assigned = assignments.set_index('company_id').profile_id
    rows = []
    titles = {p['profile_id']: p['title'] for p in profiles}
    prospects = lookalikes.sort_values('rank').copy()
    prospects['profile_rank'] = prospects.groupby('profile_id').cumcount() + 1
    for row in prospects.itertuples():
        venue = info.loc[row.company_id]
        rows.append({'profile_id': row.profile_id, 'title': titles.get(row.profile_id, ''), 'role': 'prospect',
                     'vat': venue.vat, 'company_id': row.company_id, 'company_name': venue.company_name_el,
                     'city': venue.city, 'municipality': venue.municipality_name, 'rank': int(row.rank),
                     'profile_rank': int(row.profile_rank), 'score': float(row.score),
                     'within_profile': bool(assigned.get(row.company_id) == row.profile_id),
                     'rationale': row.rationale})
    vats, labels = np.asarray(fit['vats']), np.asarray(fit['labels'])
    ids = companies.company_id.to_numpy()[fit['indices']]
    for j, profile in enumerate(profiles):
        seen = set()
        for vat, company_id in sorted(zip(vats[labels == j], ids[labels == j])):
            if vat in seen:  # one row per ΑΦΜ, even for customers with several branches in the profile
                continue
            seen.add(vat)
            venue = info.loc[company_id]
            rows.append({'profile_id': profile['profile_id'], 'title': profile['title'], 'role': 'customer',
                         'vat': vat, 'company_id': company_id, 'company_name': venue.company_name_el,
                         'city': venue.city, 'municipality': venue.municipality_name, 'rank': None,
                         'profile_rank': None, 'score': None, 'within_profile': True, 'rationale': ''})
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def clusters_json(profiles, table):
    out = []
    for p in profiles:
        part = table[table.profile_id == p['profile_id']]
        prospects = part[part.role == 'prospect']
        out.append({'profile_id': p['profile_id'], 'title': p['title'], 'description': p['description'],
                    'ai_generated': p.get('ai_generated'), 'customer_count': int((part.role == 'customer').sum()),
                    'prospect_count': len(prospects), 'prospects_within_profile': int(prospects.within_profile.sum()),
                    'customers': part[part.role == 'customer'].vat.tolist(),
                    'prospects': [{'vat': r.vat, 'company_id': r.company_id, 'company_name': r.company_name,
                                   'rank': int(r.rank), 'profile_rank': int(r.profile_rank), 'score': round(float(r.score), 6),
                                   'within_profile': bool(r.within_profile)} for r in prospects.itertuples()]})
    return out


def write_xlsx(profiles, table, path: Path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    blue, white = PatternFill('solid', fgColor='2563EB'), Font(bold=True, color='FFFFFF')
    wb = Workbook()
    summary = wb.active
    summary.title = 'Περίληψη'
    summary.append(['Profile', 'Τίτλος', 'Περιγραφή (κείμενο AI, τοπικό μοντέλο)', 'Πελάτες (ΑΦΜ)',
                    'Προτάσεις εντός profile', 'Προτάσεις συνολικά'])
    for p in profiles:
        part = table[table.profile_id == p['profile_id']]
        prospects = part[part.role == 'prospect']
        summary.append([p['profile_id'], p['title'], p['description'], int((part.role == 'customer').sum()),
                        int(prospects.within_profile.sum()), len(prospects)])
    headers = ['Κατάταξη στο profile', 'Συνολική κατάταξη', 'ΑΦΜ', 'Επωνυμία', 'Πόλη', 'Δήμος',
               'Ταίριασμα (%)', 'Εντός profile', 'Γιατί προτείνεται']
    for p in profiles:
        sheet = wb.create_sheet(p['profile_id'])
        sheet.append([f"{p['profile_id']} · {p['title']}"])
        sheet.append([p['description']])
        sheet.append(['Προτεινόμενες νέες επιχειρήσεις, από την πιο ταιριαστή. «Όχι» στο «Εντός profile» = διερευνητική πρόταση.'])
        sheet.append(headers)
        part = table[(table.profile_id == p['profile_id']) & (table.role == 'prospect')]
        for r in part.itertuples():
            sheet.append([r.profile_rank, r.rank, r.vat, r.company_name, r.city, r.municipality,
                          round(100 * r.score, 1), 'Ναι' if r.within_profile else 'Όχι', r.rationale])
        sheet['A1'].font = Font(bold=True, size=13)
        sheet['A2'].alignment = Alignment(wrap_text=True, vertical='top')
        sheet.merge_cells('A2:I2')
        sheet.row_dimensions[2].height = 120
        for cell in sheet[4]:
            cell.fill, cell.font = blue, white
        sheet.freeze_panes = 'A5'
        sheet.auto_filter.ref = f'A4:I{sheet.max_row}'
    customers = wb.create_sheet('Πελάτες ανά profile')
    customers.append(['Profile', 'Τίτλος', 'ΑΦΜ', 'Επωνυμία', 'Πόλη', 'Δήμος'])
    for r in table[table.role == 'customer'].itertuples():
        customers.append([r.profile_id, r.title, r.vat, r.company_name, r.city, r.municipality])
    for sheet in wb.worksheets:
        header_row = 4 if sheet.title not in ('Περίληψη', 'Πελάτες ανά profile') else 1
        if header_row == 1:
            for cell in sheet[1]:
                cell.fill, cell.font = blue, white
            sheet.freeze_panes = 'A2'
        for row in sheet.iter_rows(min_row=header_row + 1):
            for cell in row:
                if isinstance(cell.value, str) and cell.value.isdigit() and len(cell.value) == 9:
                    cell.number_format = '@'  # ΑΦΜ stays text, leading zeros kept
        widths = {'Περίληψη': [10, 40, 90, 14, 20, 18]}.get(sheet.title, [12, 12, 13, 36, 18, 28, 13, 12, 90])
        for i, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(i)].width = width
    for row in summary.iter_rows(min_row=2):
        row[2].alignment = Alignment(wrap_text=True, vertical='top')
    # Fixed document dates so the same analysis gives the same workbook metadata.
    wb.properties.created = wb.properties.modified = datetime(2026, 11, 1)
    wb.save(path)


def export_clusters(result, directory: Path):
    import json
    table = result['clusters']
    (directory / 'clusters.json').write_text(
        json.dumps(clusters_json(result['profiles'], table), ensure_ascii=False, indent=2), 'utf-8')
    table.to_csv(directory / 'clusters.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
    write_xlsx(result['profiles'], table, directory / 'clusters.xlsx')
