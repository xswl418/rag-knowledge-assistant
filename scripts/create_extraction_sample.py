"""Generate a controlled fixture for checking extraction across batches."""

from pathlib import Path


SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "samples"
    / "extraction_long_test.txt"
)


def build_sample_text():
    beginning = (
        "长文提取合成样例\n"
        "本文件为人工构造的流程测试资料，不是真实业务规定。\n"
        "一、资料甲\n"
        "借阅上限：3本。\n"
        "借阅期限：12天。\n\n"
        "二、档案目录\n"
    )
    entries = [
        (
            f"Catalog entry {index:03d}: "
            "The archive contains a shelf map, a reading room sketch, "
            "an inventory note, a cover description, and a storage label. "
            "These catalog descriptions concern the physical arrangement "
            "of archived materials and contain no lending policy."
        )
        for index in range(1, 61)
    ]
    ending = (
        "\n\n三、资料乙\n"
        "借阅上限：5本。\n"
        "借阅期限：12天。\n"
    )
    return beginning + "\n\n".join(entries) + ending


def main():
    text = build_sample_text()
    if SAMPLE_PATH.exists():
        if SAMPLE_PATH.read_text(encoding="utf-8") != text:
            raise SystemExit("The sample has local changes; leaving it untouched.")
    else:
        SAMPLE_PATH.write_text(text, encoding="utf-8")
    print(f"Sample: {SAMPLE_PATH.name}; characters: {len(text)}")


if __name__ == "__main__":
    main()
