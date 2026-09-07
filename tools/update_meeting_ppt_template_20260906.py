"""Update the 2026-08-30 deck in-place, preserving its original layout."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "张明远-2026.08.30.pptx"
OUTPUT = ROOT / "张明远-2026.09.06.pptx"
ASSETS = ROOT / "outputs" / "ppt_20260906_assets"


def set_paragraphs(shape, lines):
    """Replace paragraph strings while retaining the template's paragraph/run styles."""
    tf = shape.text_frame
    old = list(tf.paragraphs)
    while len(tf.paragraphs) < len(lines):
        source = old[min(len(tf.paragraphs), len(old) - 1)]
        target = tf.add_paragraph()
        target._p.get_or_add_pPr().attrib.update(source._p.get_or_add_pPr().attrib)
    for i, paragraph in enumerate(tf.paragraphs):
        text = lines[i] if i < len(lines) else ""
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run().text = text


def set_text(shape, text):
    set_paragraphs(shape, text.split("\n"))


def remove_shape(shape):
    shape._element.getparent().remove(shape._element)


def replace_visual(slide, index, path):
    old = slide.shapes[index]
    left, top, width, height = old.left, old.top, old.width, old.height
    remove_shape(old)
    slide.shapes.add_picture(str(path), left, top, width, height)


def make_collage():
    paths = [ASSETS / "stack_mid.png", ASSETS / "push_mid.png", ASSETS / "nut_mid.png"]
    labels = ["Stack  95/100", "Push  100/100", "NutAssembly  100/100"]
    canvas = Image.new("RGB", (1500, 920), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = Image.open(path).convert("RGB").resize((460, 690))
        x = 30 + i * 500
        canvas.paste(image, (x, 80))
        draw.rectangle((x, 770, x + 460, 865), fill=(23, 58, 138) if i == 0 else (27, 111, 42))
        draw.text((x + 30, 795), label, fill="white", stroke_width=0)
    out = ASSETS / "three_tasks_collage.png"
    canvas.save(out)
    return out


def main():
    prs = Presentation(SOURCE)
    collage = make_collage()

    # Cover: retain logo, banner, shapes, and all typography.
    set_text(prs.slides[0].shapes[2], "2026年09月06日\n组会进展汇报")

    # Slide 2: weekly headline and five task status cards.
    s = prs.slides[1]
    set_text(s.shapes[2], "进展一：多任务仿真示教生成与全量验证")
    set_text(s.shapes[3], "：本周总体进展")
    for idx, text in zip(
        range(4, 9),
        [
            "抬升 Lift\n100 / 100",
            "搬运放置\nPickPlace 100 / 100",
            "堆叠 Stack\n95 / 100",
            "推物 Push\n100 / 100",
            "螺母装配\n100 / 100",
        ],
    ):
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[9],
        [
            "面向 RM75B + 6 驱动 Dex Hand，推进 Top100 多任务示教生成。",
            "已有基线：",
            "Lift、PickPlace 均已完成 100 / 100。",
            "",
            "本周新增：",
            "Push v8 与 NutAssembly v22 均达到 100 / 100；Stack v18 达到 95 / 100。",
            "当前重点：",
            "收敛 Stack 最后 5 个极端几何对象。",
        ],
    )

    # Slide 3: prior state -> current state.
    s = prs.slides[2]
    set_text(s.shapes[2], "进展一：多任务仿真示教生成与全量验证")
    set_text(s.shapes[3], "：上周目标与本周结果")
    for idx, text in zip(
        range(4, 10),
        [
            "4 项任务\n待推进",
            "2 项任务\n本周满分",
            "Stack\n95%",
            "目标设定\n→\n全量执行",
            "执行成功\n→\n稳定复验",
            "问题收敛：从通用执行链路到 Stack 末端稳定",
        ],
    ):
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[10],
        [
            "8.30 阶段目标：",
            "在 Lift / PickPlace 基础上推进 Stack、Push、NutAssembly。",
            "本周正式结果：",
            "Push 100 / 100；NutAssembly 100 / 100；Stack 95 / 100。",
            "验证升级：",
            "全部结果以 C MuJoCo 完整执行与持续稳定复验为准。",
            "当前判断：最后瓶颈集中于 Stack 释放后的落姿稳定。",
        ],
    )

    # Slide 4: unified pipeline.
    s = prs.slides[3]
    set_text(s.shapes[2], "进展一：多任务仿真示教生成与全量验证")
    set_text(s.shapes[3], "：统一任务生成与批处理流程")
    for idx, text in {
        4: "候选\n搜索",
        5: "C MuJoCo\n执行",
        6: "任务生成",
        8: "稳定性\n复验",
        9: "轨迹\n报告",
        10: "Top100\n全量",
        11: "断点续跑 + 逐对象日志 + 失败阶段诊断",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[12],
        [
            "统一入口：",
            "同一 Top100 选择集，三项任务分别运行候选搜索与物理执行。",
            "统一产物：",
            "逐对象 manifest、episode.npz、运行日志和全量 summary.json。",
            "统一验证：",
            "成功必须同时满足可执行、任务完成和持续物理稳定。",
            "工程能力：",
            "支持并行批处理、断点续跑和失败对象定向重跑。",
        ],
    )

    # Slide 5: same full-page visual slot, new weekly overview.
    s = prs.slides[4]
    set_text(s.shapes[2], "进展一：多任务仿真示教生成与全量验证")
    set_text(s.shapes[3], "：三项任务代表性执行画面")
    replace_visual(s, 4, collage)

    # Slide 6: Stack mechanics.
    s = prs.slides[5]
    set_text(s.shapes[2], "进展二：Stack 无斜坡悬空搬运")
    set_text(s.shapes[3], "：执行约束与放置流程")
    for idx, text in {
        4: "Lift\n抓持前缀",
        6: "悬空\n横向运输",
        8: "冻结 XY\n垂直下降",
        10: "承重\n释放",
        11: "硬约束\n无斜坡 · 运输零平台接触 · 纯垂直放置",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[12],
        [
            "场景约束：",
            "使用固定焊接实体平台，不引入斜坡、托架或隐式支撑。",
            "运输约束：",
            "平台沿抓持方向搜索；搬运阶段逐帧检查，禁止接触平台。",
            "放置约束：",
            "进入落点阈值后冻结 XY，只允许垂直下降。",
            "稳定验证：",
            "承重释放、后退 100 帧，再连续复验 150 帧。",
        ],
    )

    # Slide 7: Stack validation signals.
    s = prs.slides[6]
    set_text(s.shapes[2], "进展二：Stack 无斜坡悬空搬运")
    set_text(s.shapes[3], "：严格成功判据")
    for idx, text in {
        4: "悬空\n95",
        5: "纯垂直\n95",
        6: "驻留\n95",
        7: "运输接触 0",
        8: "后退 100 帧",
        9: "复验 150 帧",
        10: "平台稳定",
        11: "逐帧接触审计 + 任务成功驻留",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[12],
        [
            "悬空运输：",
            "所有 transport 帧中，目标物体不得接触支撑平台。",
            "纯垂直放置：",
            "到达平台上方后冻结水平落点，仅沿 Z 轴下降。",
            "释放复验：",
            "机械手完全退让后，目标物体需在平台上持续稳定。",
            "结果意义：",
            "95 条轨迹均满足 suspended_transport=True 与 150 帧复验。",
        ],
    )

    # Slide 8: Stack result.
    s = prs.slides[7]
    set_text(s.shapes[2], "进展二：Stack 无斜坡悬空搬运")
    set_text(s.shapes[3], "：Top100 正式批次结果")
    for idx, text in {
        4: "候选方向",
        6: "抓持模式",
        8: "垂直放置",
        10: "稳定释放",
        12: "平台接触反馈",
        14: "C MuJoCo",
        15: "主线变化：失败已从抓取 / 运输收敛到落点与释放后的姿态稳定",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[16],
        [
            "正式结果：",
            "Stack v18 = 95 / 100。",
            "成功对象：",
            "包含 lemon、orange、plum、Saccharomyces 等历史难例。",
            "剩余失败：",
            "EGAD E0、玩具冰箱、胶带、染发凝胶、尿布桶补充装。",
            "共同原因：",
            "释放后未能在支撑平台上保持稳定，而非运输接触作弊。",
        ],
    )

    # Slide 9: Push headline.
    s = prs.slides[8]
    set_text(s.shapes[2], "进展三：Push 掌心皮肤触觉闭环")
    set_text(s.shapes[3], "：Top100 全量结果")
    for idx, text in {
        4: "Push Top100\n100 / 100",
        5: "触觉闭环\n100",
        6: "目标复验\n100",
        7: "GSO\n75/75",
        8: "YCB\n20/20",
        9: "EGAD\n5/5",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[10],
        [
            "最终结果：",
            "掌心皮肤触觉闭环 Push 达到 100 / 100。",
            "关键设计：",
            "低推动高度、掌心 taxel 建联、方向自适应与接触丢失重捕获。",
            "闭环信号：",
            "融合触觉、接触距离、目标距离趋势与横向偏差。",
            "末端复验：",
            "进入目标区域后执行反馈式微推并持续验证稳定性。",
            "版本：",
            "palm-tactile-push-v8。",
        ],
    )

    # Slide 10: Push convergence.
    s = prs.slides[9]
    set_text(s.shapes[2], "进展三：Push 掌心皮肤触觉闭环")
    set_text(s.shapes[3], "：困难物体分层恢复")
    for idx, text in {
        4: "接近\n建联",
        6: "闭环\n推进",
        8: "丢失\n重捕获",
        10: "100\n复验",
        11: "掌心盲区\n→ 边缘覆盖",
        12: "滚动偏航\n→ 自适应转向",
        13: "100 / 100 难例无需改变任务判据",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[14],
        [
            "主要问题：",
            "掌心边缘触觉盲区、球形物体滚偏、接触丢失后继续空推。",
            "触觉修正：",
            "扩大有效 palm taxel 覆盖，并降低推动高度以减小倾覆力矩。",
            "运动修正：",
            "根据横向误差自适应转向，失联时主动回到接触面。",
            "最终结果：",
            "所有 Top100 对象通过相同目标区域与稳定性判据。",
        ],
    )

    # Slide 11: preserve the original large media slot, replace only its content.
    s = prs.slides[10]
    set_text(s.shapes[2], "进展三：Push 掌心皮肤触觉闭环")
    set_text(s.shapes[3], "：代表性闭环推动执行")
    replace_visual(s, 4, ASSETS / "push_mid.png")

    # Slide 12: task relationship.
    s = prs.slides[11]
    set_text(s.shapes[3], "进展四：NutAssembly 固定螺母插入")
    set_text(s.shapes[4], "：任务结构与技能迁移")
    set_paragraphs(
        s.shapes[0],
        [
            "任务设定：",
            "螺母固定在桌面，欠驱动手抓持带把手螺栓并完成插入。",
            "物理结构：",
            "保留真实壁面碰撞、形状配合和被动角度顺应。",
            "变体空间：",
            "圆形、方形及 3–16 边形，共 10 个尺寸族、100 个可复现变体。",
            "成功判据：",
            "位置、竖直度、插入深度、低速度与连续稳定帧同时满足。",
        ],
    )
    set_text(s.shapes[5], "固定螺母 + 自由螺栓 + 真实壁面碰撞")
    for idx, text in {
        6: "专用抓持",
        7: "抬升",
        8: "对孔运输",
        9: "竖直插入",
        10: "稳定驻留",
        11: "形状配合\n3–16 边形",
        12: "欠驱动顺应\n6 驱动",
        13: "程序化变体\n100 组",
    }.items():
        set_text(s.shapes[idx], text)

    # Slide 13: Nut execution.
    s = prs.slides[12]
    set_text(s.shapes[2], "进展四：NutAssembly 固定螺母插入")
    set_text(s.shapes[3], "：欠驱动抓持与插入流程")
    for idx, text in {
        4: "专用抓持",
        6: "抬升",
        8: "对孔运输",
        9: "竖直插入",
        10: "驻留复验",
        11: "位置 + 竖直度 + 深度 + 低速\n连续 20 帧 = 成功",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[12],
        [
            "抓持对象：",
            "欠驱动六执行器手抓持螺栓上部公共把手。",
            "搬运策略：",
            "先抬升保持，再将螺栓轴线运输到固定螺母中心。",
            "插入策略：",
            "垂直下降，并依靠宽松导向配合吸收被动倾角误差。",
            "成功验证：",
            "满足几何与速度阈值后连续驻留 20 帧。",
        ],
    )

    # Slide 14: Nut full batch.
    s = prs.slides[13]
    set_text(s.shapes[3], "进展四：NutAssembly 固定螺母插入")
    set_text(s.shapes[4], "：100 个程序化变体全量结果")
    set_text(s.shapes[5], "NutAssembly Top100\n100 / 100")
    set_text(s.shapes[6], "10\n形状截面")
    set_text(s.shapes[7], "10\n尺寸族")
    set_text(s.shapes[8], "所有变体通过同一物理成功与持续稳定判据")
    set_paragraphs(
        s.shapes[0],
        [
            "正式结果：",
            "fixed-nut-bolt-transfer-v22 = 100 / 100。",
            "变体组成：",
            "10 种截面族 × 10 种尺寸族，覆盖圆、方及规则多边形。",
            "困难变体：",
            "三角棱柱限制外接半径，保持公共把手的可抓持性。",
            "稳定性处理：",
            "慢收敛变体延长 settle，但不改变最终物理成功判据。",
            "结论：",
            "欠驱动抓持与导向配合可覆盖全部 100 个程序化装配变体。",
        ],
    )

    # Slide 15: current status and next actions.
    s = prs.slides[14]
    set_text(s.shapes[3], "进展五：当前状态与后续推进")
    set_text(s.shapes[4], "：三项任务收敛情况")
    for idx, text in {
        5: "Lift\n100%",
        7: "PickPlace\n100%",
        9: "Stack\n95%",
        10: "Push\n100%",
        11: "NutAssembly\n100%",
        12: "当前重点：Stack 最后 5 个对象的稳定落姿与柔顺释放\n完成后统一演示、测试、文档、提交与发布",
    }.items():
        set_text(s.shapes[idx], text)
    set_paragraphs(
        s.shapes[0],
        [
            "本周已完成：",
            "Push 100 / 100；NutAssembly 100 / 100；Stack 95 / 100。",
            "Stack 剩余机制：",
            "极端几何对象在落地或张手后翻滚 / 回弹，稳定域过窄。",
            "下一步修正：",
            "按质心与支撑多边形生成落姿，加入平台接触微调。",
            "保持约束：",
            "不降低无斜坡、真悬空、纯垂直放置与稳定复验标准。",
            "最终目标：",
            "Stack 补齐至 100 / 100，三项任务统一整理发布。",
        ],
    )

    # Slide 16: preserve the original media dimensions for NutAssembly visual.
    s = prs.slides[15]
    set_text(s.shapes[1], "进展四：NutAssembly 固定螺母插入")
    set_text(s.shapes[2], "：代表性插入与稳定驻留")
    replace_visual(s, 3, ASSETS / "nut_mid.png")

    # Closing slide.
    set_text(prs.slides[16].shapes[3], "哈尔滨工业大学\n2026年09月06日")

    # The template's status/process boxes use light fills. Force their text to
    # dark ink after replacement so inherited white run colors cannot hide it.
    dark = RGBColor(25, 27, 31)
    for slide_index, slide in enumerate(prs.slides):
        if slide_index == 0 or slide_index == len(prs.slides) - 1:
            continue
        for shape in slide.shapes:
            if shape.shape_type != MSO_SHAPE_TYPE.AUTO_SHAPE or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = dark

    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
