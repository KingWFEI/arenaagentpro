import ast
import json
import re


def extract_json_from_text(text):
    # 正则表达式匹配JSON字符串(数组或对象)
    json_pattern = r"\[\s*{.*?}\s*\]"
    try:
        # 替换单引号为双引号，确保符合JSON格式
        text = text.replace("'", "_")
        text = text.replace("True", "true")
        text = text.replace("False", "false")

        # 找到JSON字符串
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group()
            # 将JSON字符串解析为Python对象
            json_data = json.loads(json_str)
            return json_data
        else:
            print("未找到符合条件的JSON数据")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

def extract_json_with_eval(text):
    try:
        json_pattern = r"\[\s*{.*?}\s*\]"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if not matches:
            print("未找到符合条件的JSON数据")
            return None

        # 获取最后一个匹配
        json_str = matches[-1]

        # 使用eval()直接将Python格式的字符串转换为Python对象
        # 注意：仅在数据来源可信的情况下使用eval()
        python_obj = eval(json_str)

        # 如果需要标准JSON输出，可以再次转换
        return python_obj

    except Exception as e:
        print(f"处理错误: {e}")
        return None


def fix_structural_dicts_only(text):
    """
    仅修复 JSON 字段值为 Python dict 的结构中的单引号
    不修改字符串内部的自然语言部分
    """
    # 匹配所有字段形如："xxx": {'key': val, ...}
    def replace_match(m):
        key = m.group(1)
        dict_str = m.group(2)
        # 替换 dict 内的结构性单引号
        fixed_dict = re.sub(r"'([^']+)'\s*:", r'"\1":', dict_str)  # 键
        fixed_dict = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed_dict)  # 值
        return f'"{key}": {fixed_dict}'

    # 修复字段为 dict 的结构
    pattern = r'"(\w+)":\s*({[^{}]+})'
    fixed = re.sub(pattern, replace_match, text)
    return fixed


def balanced_spans(text: str, opener: str, closer: str) -> list[str]:
    """扫描出所有括号配对的片段，跳过字符串字面量内部的括号。

    不能用 r"\\[\\s*{.*?}\\s*\\]" 这类懒惰正则：回答里只要出现"对象里再套一个
    数组"（例如 scene_annotations），它就会在内层数组的 ] 处提前收尾，截出的
    片段缺少外层的 } 和 ]，解析必然失败。
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            if depth == 0:
                start = index
            depth += 1
        elif char == closer and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
                start = -1
    return spans


def extract_last_json_from_text(text):
    """
    提取最后一个 JSON 片段（数组或对象）。
    - 若整体就是合法 JSON，直接解析。
    - 否则按括号配对扫描候选片段，从后往前取第一个能解析的。
    """
    stripped = (text or "").strip()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except Exception:
            pass

    try:
        text = text.replace("True", "true").replace("False", "false").replace("None", "null")
        # 处理常见中文符号
        text = (
            text.replace("，", ",")
            .replace("：", ":")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("；", ";")
            .replace("【", "[")
            .replace("】", "]")
            .replace("（", "(")
            .replace("）", ")")
            .replace("。", ".")
            .replace("？", "?")
            .replace("！", "!")
        )

        # 数组优先（回答约定是长度为 1 的数组），再退回到裸对象；各自从后往前，
        # 最后一段才是本次回答。顺序不能颠倒，否则会先命中数组里的某个内层对象。
        candidates = list(reversed(balanced_spans(text, "[", "]")))
        candidates += list(reversed(balanced_spans(text, "{", "}")))
        for candidate in candidates:
            try:
                return json.loads(fix_structural_dicts_only(candidate))
            except json.JSONDecodeError:
                continue
        print("未找到符合条件的JSON数据")
        return "未找到符合条件的JSON数据"
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return f"JSON解析错误: {e}"


if __name__ == "__main__":
    # 示例文本
    text_with_json = """
    这是一些其他的文本，不包含JSON数据。
[
  {
    "think": "我拿到了黑色杯子，我将它放到餐桌上。餐桌位置在{'X': -348.715576171875, 'Y': 97.6534194946289, 'Z': 3.751265048980713}",
    "action": "put_down_sth_to_location",
    "parameters": {
      "location": {'X': -348.715576171875, 'Y': 97.6534194946289, 'Z': 63.751265048980713}
    },
    "output": 0
  }
]
    """
    # 使用eval提取数据
    parsed_data = extract_last_json_from_text(text_with_json)
    if parsed_data:
        print("解析出的数据:")
        # 转换为标准JSON输出
        print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
