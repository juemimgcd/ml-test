import os
from pathlib import Path
import mammoth

def convert_docx_to_md(docx_path, md_path):
    """
    使用 mammoth 将 docx 转换为 md
    """
    try:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(result.value)
            
            # 打印警告信息（如果有）
            if result.messages:
                print(f"⚠️ 转换 {os.path.basename(docx_path)} 时出现警告:")
                for message in result.messages:
                    print(f"   - {message}")
            
            print(f"✅ 成功转换: {os.path.basename(docx_path)} -> {os.path.basename(md_path)}")
            return True
    except Exception as e:
        print(f"❌ 转换失败 {os.path.basename(docx_path)}: {e}")
        return False

def main():
    papers_dir = Path("papers")
    
    if not papers_dir.exists():
        print("找不到 papers 目录")
        return

    docx_files = list(papers_dir.glob("*.docx"))
    
    if not docx_files:
        print("papers 目录下没有找到 .docx 文件")
        return

    print(f"找到 {len(docx_files)} 个 .docx 文件，开始转换...")

    for docx_file in docx_files:
        md_file = docx_file.with_suffix('.md')
        convert_docx_to_md(str(docx_file), str(md_file))

if __name__ == "__main__":
    main()
