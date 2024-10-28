from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@register.simple_tag
def render_table(table_class, table_data):
    headers = table_data["headers"]
    rows = table_data["rows"]
    links = table_data["links"]

    html = f'<table class="{table_class}">'
    html += "<thead><tr>"
    for header in headers:
        html += f"<th>{header}</th>"
    html += "</tr></thead><tbody>"

    for row, row_links in zip(rows, links):
        html += "<tr>"
        for header in headers:
            value = row.get(header, "")
            if header in row_links:
                url = row_links[header]
                html += f'<td><a href="{url}">{value}</a></td>'
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return mark_safe(html)
