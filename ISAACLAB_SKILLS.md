# Isaac Lab Agent Skills

## 来源与安装方式

- 用户提供的 develop/skills 页面当前跳转到 main/skills。
- 上游仓库： https://github.com/isaac-sim/IsaacLab
- 固定提交： a9fb50e5388724a680bbc5c1f836917271605a99
- 共 24 个技能：7 个 developer 技能，17 个 user 技能。
- 上游 skills/README.md 要求保留仓库目录结构，禁止把技能平铺复制到全局技能目录，以免相对引用断开。
- 本项目安装器将源码及引用材料保留在 CODEX_HOME/skill-sources/ 下（CODEX_HOME 默认是用户主目录下的 .codex），并在用户主目录的 .agents/skills/ 下按 frontmatter 名称建立相对符号链接。
- 这些是给智能体使用的指导文件，不是 Isaac Lab 仿真或训练运行时安装包。

## 换主机重装

在项目根目录执行，需要 Python 3、Git、GitHub 网络访问和用户目录写权限：

    python3 scripts/install_isaaclab_skills.py

安装器固定源码版本，保留已有的不相关技能；发现同名冲突时停止，不覆盖。重复执行会检查并复用正确的安装。不要手动复制指向旧主机的链接；在新主机运行上述命令。

## 使用

从下一轮对话开始应能发现已安装技能。若当前客户端未刷新，重新打开会话。
可直接要求使用 isaaclab-migrating-from-isaac-gym、isaaclab-building-environments 或 isaaclab-training-rl-agents 等实际技能名称。

技能来自上游 main，包含 Isaac Lab 3.x、多后端等内容。执行迁移和训练时仍须核实项目实际安装的 Isaac Lab 版本，不能直接把上游 main 的 API 当成本项目已支持的 API。

## 验证状态

- 已确认固定提交及 24 个 SKILL.md 入口。
- 安装器静态语法检查通过。
- 用户级安装完成：24 个技能入口均能读取 SKILL.md，全部链接指向持久化的固定提交目录。
- 在安装后的源码目录运行 python3 tools/skills/cli.py check，通过：Validated 24 skills.
- git diff --check 通过。安装脚本已在本机使用预下载源码成功执行；新主机的全新网络下载路径尚未单独实测。
- 本次不执行机器人训练，不代表现有 G1/CMoE 迁移已完成。
