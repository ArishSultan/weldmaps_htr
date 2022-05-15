def parse_jacobs(table_ocr_data):
    tables = []
    table = None

    now_is_header = False
    current_columns = None
    for box in table_ocr_data:
        txt = combine_simple_text(box['text'])

        if now_is_header:
            # Reset the flag and read columns.
            now_is_header = False
            current_columns = parse_jacobs_columns(box)

        elif "*" in txt:
            if table is not None:
                tables.append(table)
            table = []

            # Obviously next line would be header
            now_is_header = True

        else:
            # This will be a row, but we also need to filter non rows i.e. sub headers etc.
            row_entry = {}
            row = filter_row(box)

            # Check if this a New Row, Continuation or an Ignorable Header
            if row[1][0] >= current_columns[0][1] - 5 and row[1][0] + row[2][0] <= current_columns[0][2]:
                # This is a new Row.
                for i in range(len(row[0])):
                    txt = row[0][i]
                    left = row[1][i]
                    width = left + row[2][i]

                    for j in range(len(current_columns)):
                        if left >= current_columns[j][1] - 5 and width <= current_columns[j][2]:
                            if current_columns[j][0] in row_entry:
                                row_entry[current_columns[j][0]] += ' ' + txt
                            else:
                                row_entry[current_columns[j][0]] = txt
            elif row[1][0] >= current_columns[1][1] + 20:
                for i in range(len(row[0])):
                    table[len(table) - 1][current_columns[1][0]] += ' ' + row[0][i]

            if len(row_entry) > 0:
                table.append(row_entry)

    if len(table) > 0:
        tables.append(table)

    return tables


def combine_simple_text(text_list):
    chunk = ''
    for t in text_list:
        if t != '':
            chunk += ' ' + t

    return chunk.strip()


def filter_row(box):
    conf = box['conf']
    count = 0
    for i in range(len(conf)):
        if conf[i] != '-1':
            count = i
            break

    return box['text'][count:], box['left'][count:], box['width'][count:]


def parse_jacobs_columns(box):
    columns = []

    text = box['text']
    conf = box['conf']
    left = box['left']
    width = box['width']

    j = 0
    for i in range(len(conf)):
        if conf[i] == '-1':
            continue

        if j == 4:
            txt = ''.join(text[i:])
            columns.append((txt, left[i], width[0]))
            break
        else:
            txt = text[i]
            columns.append((txt, left[i], left[i + 1] - 5))

        j += 1

    return columns
