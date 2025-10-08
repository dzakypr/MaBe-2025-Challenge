import ast
def extract_tracked_body_parts(df):
    body_parts_tracked_str = df['body_parts_tracked'].unique()
    body_parts_tracked_arr = []
    for i in body_parts_tracked_str:
        body_parts_tracked_arr.append(ast.literal_eval(i))
    body_parts_tracked_list = []
    for arr in body_parts_tracked_arr:
        for i in arr:
            if i not in body_parts_tracked_list:
                body_parts_tracked_list.append(i)
    return body_parts_tracked_list

