"""求职信工作流的状态定义；业务规则见 state.md。

TypedDict 提供类型声明，不执行运行时校验；输入和模型输出由对应节点校验。
"""

from typing import Final, Literal, TypedDict

try:
    from . import template
except ImportError:
    from . import template_example as template


MAX_HUMAN_INPUTS: Final = 2


class JDAnalysis(TypedDict):
    """仅记录 JD 中有依据的信息，未提供的信息使用 None 或空列表。"""

    job_title: str | None
    company: str | None
    summary: str
    responsibilities: list[str]
    requirements: list[str]
    nice_to_haves: list[str]


class Evaluation(TypedDict):
    """仅评估是否需要额外材料；总结和问题使用中文，ready 时 questions 为空。"""

    status: Literal["ready", "missing_info"]
    summary: str
    questions: list[str]


class HumanInput(TypedDict):
    """保留提问快照及原始回答；空白回答表示直接基于现有事实生成。"""

    summary: str
    questions: list[str]
    answer: str


class OutputFiles(TypedDict):
    """文件成功保存后才填写绝对路径；PDF 与 Markdown 使用同一正文。"""

    markdown_path: str | None
    pdf_path: str | None


class AgentState(TypedDict):
    """人工补充轮数取 len(human_inputs)，最终正文使用英文。

    human_inputs 不设置追加 reducer：节点更新时返回包含新记录的完整列表。
    """

    jd_text: str
    jd_analysis: JDAnalysis | None
    cv_text: str
    scenes_text: str
    template_text: str
    evaluation: Evaluation | None
    prepared_materials: str | None
    human_inputs: list[HumanInput]
    output_markdown: str | None  # generate 写入，save_output 规范空白与硬换行后回写。
    output_files: OutputFiles


def create_initial_state(
    *,
    jd_text: str,
) -> AgentState:
    """使用 JD 和预存文本初始化状态，每次调用使用独立的列表和字典。"""
    return {
        "jd_text": jd_text,
        "jd_analysis": None,
        "cv_text": template.CV_TEXT,
        "scenes_text": template.SCENES_TEXT,
        "template_text": template.COVER_LETTER_TEMPLATE,
        "evaluation": None,
        "prepared_materials": None,
        "human_inputs": [],
        "output_markdown": None,
        "output_files": {"markdown_path": None, "pdf_path": None},
    }
