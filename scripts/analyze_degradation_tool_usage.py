#!/usr/bin/env python3
"""分析训练日志中每个退化类型图片是否至少调用一次对应去退化工具。

去退化工具与退化类型的对应关系：
- fog → ridcp, kanet (去雾)
- rain_streak → turbo_rain, s2former, idt (去雨条纹)
- rain_drop → idt (去雨滴)
- rain_drive → turbo_rain, s2former, idt (去雨驱动)
- night → hvicidnet, lightdiff, retinexformer_fivek (去夜间/低光照)
- snow → turbo_snow, snowmaster (去雪)

通用工具（不属于任何特定退化类型）：
- real_esrgan, scunet (通用超分辨率/去噪)
"""

import re
import json
from collections import defaultdict
from pathlib import Path

# 退化类型 → 专用去退化工具映射
DEGRADATION_TOOL_MAP = {
    "fog": ["ridcp", "kanet"],
    "rain_streak": ["turbo_rain", "s2former", "idt"],
    "rain_drop": ["idt"],
    "rain_drive": ["turbo_rain", "s2former", "idt"],
    "night": ["hvicidnet", "lightdiff", "retinexformer_fivek"],
    "snow": ["turbo_snow", "snowmaster"],
}

# 通用工具
GENERAL_TOOLS = ["real_esrgan", "scunet"]

# 所有专用工具集合
ALL_SPECIALIZED_TOOLS = set()
for tools in DEGRADATION_TOOL_MAP.values():
    ALL_SPECIALIZED_TOOLS.update(tools)


def parse_log(log_path):
    """解析日志文件，提取每个实例的退化类型和工具调用轨迹。"""
    
    # 实例ID → {degradation, actions: [(action, step, reward)], image_path}
    instances = {}
    
    # 正则模式
    create_pattern = re.compile(
        r"Created restoration instance ([\w-]+) for: .+ \(degradation=(\w+)"
    )
    apply_pattern = re.compile(
        r"Instance ([\w-]+): applying '(\w+)'"
    )
    done_pattern = re.compile(
        r"Instance ([\w-]+): '(\w+)' done, step=(\d+), reward=([-\d.]+)"
    )
    stop_pattern = re.compile(
        r"Instance ([\w-]+): stop action at step (\d+)"
    )
    
    with open(log_path, "r") as f:
        for line in f:
            # 创建实例
            m = create_pattern.search(line)
            if m:
                instance_id = m.group(1)
                degradation = m.group(2)
                instances[instance_id] = {
                    "degradation": degradation,
                    "actions": [],
                    "image_path": None,
                }
                continue
            
            # 应用工具
            m = apply_pattern.search(line)
            if m:
                instance_id = m.group(1)
                action = m.group(2)
                if instance_id in instances:
                    instances[instance_id]["actions"].append({
                        "action": action,
                        "step": None,
                        "reward": None,
                        "type": "apply",
                    })
                continue
            
            # 工具完成
            m = done_pattern.search(line)
            if m:
                instance_id = m.group(1)
                action = m.group(2)
                step = int(m.group(3))
                reward = float(m.group(4))
                if instance_id in instances:
                    # 更新最后一个匹配的apply记录
                    for a in reversed(instances[instance_id]["actions"]):
                        if a["action"] == action and a["step"] is None:
                            a["step"] = step
                            a["reward"] = reward
                            break
                continue
            
            # stop动作
            m = stop_pattern.search(line)
            if m:
                instance_id = m.group(1)
                step = int(m.group(2))
                if instance_id in instances:
                    instances[instance_id]["actions"].append({
                        "action": "stop",
                        "step": step,
                        "reward": None,
                        "type": "stop",
                    })
                continue
    
    return instances


def analyze_instances(instances):
    """分析每个实例是否至少调用一次对应去退化工具。"""
    
    # 按退化类型分组
    by_degradation = defaultdict(list)
    for inst_id, inst_data in instances.items():
        by_degradation[inst_data["degradation"]].append((inst_id, inst_data))
    
    results = {}
    total_instances = 0
    total_with_specialized = 0
    
    for deg_type, inst_list in sorted(by_degradation.items()):
        specialized_tools = DEGRADATION_TOOL_MAP.get(deg_type, [])
        
        count = len(inst_list)
        with_specialized = 0
        with_general_only = 0  # 只用了通用工具，没用专用工具
        with_no_tool = 0  # 完全没调用任何工具（只有stop）
        
        # 详细统计
        action_stats = defaultdict(int)
        first_action_stats = defaultdict(int)  # 第一个动作统计
        
        for inst_id, inst_data in inst_list:
            actions = [a["action"] for a in inst_data["actions"]]
            
            # 是否至少调用一次专用工具
            has_specialized = any(a in specialized_tools for a in actions)
            has_general = any(a in GENERAL_TOOLS for a in actions)
            has_any_tool = any(a != "stop" for a in actions)
            
            if has_specialized:
                with_specialized += 1
            elif has_general and not has_specialized:
                with_general_only += 1
            elif not has_any_tool:
                with_no_tool += 1
            
            # 统计所有动作
            for a in actions:
                action_stats[a] += 1
            
            # 统计第一个非stop动作
            for a in actions:
                if a != "stop":
                    first_action_stats[a] += 1
                    break
        
        results[deg_type] = {
            "total": count,
            "with_specialized": with_specialized,
            "with_general_only": with_general_only,
            "with_no_tool": with_no_tool,
            "specialized_rate": with_specialized / count if count > 0 else 0,
            "specialized_tools": specialized_tools,
            "action_stats": dict(action_stats),
            "first_action_stats": dict(first_action_stats),
        }
        
        total_instances += count
        total_with_specialized += with_specialized
    
    results["_overall"] = {
        "total_instances": total_instances,
        "total_with_specialized": total_with_specialized,
        "overall_specialized_rate": total_with_specialized / total_instances if total_instances > 0 else 0,
    }
    
    return results


def print_results(results):
    """打印分析结果。"""
    
    print("=" * 80)
    print("退化类型 → 专用去退化工具调用分析")
    print("=" * 80)
    
    overall = results["_overall"]
    print(f"\n📊 总体统计:")
    print(f"  总实例数: {overall['total_instances']}")
    print(f"  至少调用一次专用工具: {overall['total_with_specialized']} ({overall['overall_specialized_rate']*100:.1f}%)")
    
    print(f"\n📋 各退化类型详细统计:")
    print("-" * 80)
    
    for deg_type in sorted([k for k in results if k != "_overall"]):
        r = results[deg_type]
        print(f"\n  🔹 {deg_type} (专用工具: {', '.join(r['specialized_tools'])})")
        print(f"    总实例数: {r['total']}")
        print(f"    至少调用一次专用工具: {r['with_specialized']} ({r['specialized_rate']*100:.1f}%)")
        print(f"    只用通用工具(没用专用): {r['with_general_only']} ({r['with_general_only']/r['total']*100:.1f}%)")
        print(f"    完全没调用工具: {r['with_no_tool']} ({r['with_no_tool']/r['total']*100:.1f}%)")
        
        print(f"    所有动作统计:")
        for action, count in sorted(r["action_stats"].items(), key=lambda x: -x[1]):
            is_specialized = action in r["specialized_tools"]
            is_general = action in GENERAL_TOOLS
            tag = "★专用" if is_specialized else ("◆通用" if is_general else "")
            pct = count / sum(r["action_stats"].values()) * 100
            print(f"      {action}: {count} ({pct:.1f}%) {tag}")
        
        print(f"    第一个动作统计:")
        for action, count in sorted(r["first_action_stats"].items(), key=lambda x: -x[1]):
            is_specialized = action in r["specialized_tools"]
            is_general = action in GENERAL_TOOLS
            tag = "★专用" if is_specialized else ("◆通用" if is_general else "")
            pct = count / sum(r["first_action_stats"].values()) * 100
            print(f"      {action}: {count} ({pct:.1f}%) {tag}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    if overall['overall_specialized_rate'] >= 0.95:
        print("✅ 几乎所有图片都至少调用了一次对应去退化工具")
    elif overall['overall_specialized_rate'] >= 0.8:
        print("⚠️ 大部分图片至少调用了一次对应去退化工具，但仍有改进空间")
    elif overall['overall_specialized_rate'] >= 0.5:
        print("❌ 只有约一半的图片至少调用了一次对应去退化工具，问题较严重")
    else:
        print("❌ 大部分图片没有调用对应去退化工具，策略塌缩严重")


if __name__ == "__main__":
    log_path = Path("/home/LXJ/Python_Projects/verl/log/restoration_tool_info.log")
    instances = parse_log(log_path)
    results = analyze_instances(instances)
    print_results(results)
    
    # 保存JSON结果
    output_path = Path("/home/LXJ/Python_Projects/verl/log/degradation_tool_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_path}")