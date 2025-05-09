import os, json
from glob import glob

# 1) 클래스 이름 리스트 정의 (LabelMe에서 쓴 순서대로)
CLASSES = ["ear", "nostril", "lips"]

def convert(json_path, out_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_w = data['imageWidth']
    img_h = data['imageHeight']

    lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASSES:
            continue
        cls_id = CLASSES.index(label)
        # LabelMe rectangle: 두 점(point[0], point[1])
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]
        # 좌표 정리
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        # YOLO normalized center/wh
        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # 한 파일에 여러 객체 가능
    if lines:
        os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
        with open(out_txt_path, 'w') as f:
            f.write("\n".join(lines))

def main():
    json_files = glob(f"./*.json")  # LabelMe JSON 원본들이 이곳에 있다 가정
    for jp in json_files:
        img_name = os.path.splitext(os.path.basename(jp))[0]
        txt_out = f"./{img_name}.txt"
        convert(jp, txt_out)

if __name__ == "__main__":
    main()
