# Goal Management Visualization Guide

This guide illustrates the various visualization tools available in the goal management CLI.

## Text-Based Visualizations

### Goal Tree (`goal tree`)

The `goal tree` command shows the hierarchical structure of your goals and subgoals with status indicators:

```
Goal Tree:
└── ✅ G1: Implement authentication system
    ├── ✅ S1: Implement user registration
    │   └── 🔘 S4: Design registration form
    ├── 🟠 S2: Implement login/logout
    │   ├── ✅ S5: Create login form
    │   └── ✅ S6: Implement session management
    └── ⚪ S3: Implement password reset
        └── 🔘 S7: Design password reset flow

Status Legend:
✅ Complete
🟠 All subgoals complete (ready to merge)
⚪ Incomplete (some subgoals pending)
🔘 No subgoals
```

### Goal Status (`goal status`)

The `goal status` command provides a detailed view of the completion status with metadata:

```
Goal Status:
✅ G1: Implement authentication system
   Completed: 20250401_143022
   Merged: S1 at 20250401_142800
   Merged: S2 at 20250401_143000
  ✅ S1: Implement user registration
     Completed: 20250330_091522
    🔘 S4: Design registration form
  🟠 S2: Implement login/logout
     Completed: 20250331_161034
    ✅ S5: Create login form
       Completed: 20250331_142210
    ✅ S6: Implement session management
       Completed: 20250331_155522
  ⚪ S3: Implement password reset
    🔘 S7: Design password reset flow

Status Legend:
✅ Complete
🟠 All subgoals complete (ready to merge)
⚪ Incomplete (some subgoals pending)
🔘 No subgoals
```

### Goal History (`goal history`)

The `goal history` command shows a chronological timeline of your goals:

```
Goal History:
=============
[2025-03-28 10:15:22] G1: Implement authentication system

[2025-03-28 14:30:45] S1: Implement user registration
  └── Subgoal of: G1

[2025-03-28 16:20:11] S2: Implement login/logout
  └── Subgoal of: G1

[2025-03-29 09:45:33] S3: Implement password reset
  └── Subgoal of: G1

[2025-03-29 11:22:18] S4: Design registration form
  └── Subgoal of: S1

[2025-03-30 09:15:22] S1: Implement user registration
  └── Completed: 2025-03-30 09:15:22

[2025-03-30 14:25:37] S5: Create login form
  └── Subgoal of: S2

[2025-03-30 15:40:22] S6: Implement session management
  └── Subgoal of: S2

[2025-03-31 14:22:10] S5: Create login form
  └── Completed: 2025-03-31 14:22:10

[2025-03-31 15:55:22] S6: Implement session management
  └── Completed: 2025-03-31 15:55:22

[2025-03-31 16:10:34] S2: Implement login/logout
  └── Completed: 2025-03-31 16:10:34

[2025-04-01 09:22:15] S7: Design password reset flow
  └── Subgoal of: S3

[2025-04-01 14:28:00] G1: Implement authentication system
  └── Merged: S1 at 2025-04-01 14:28:00

[2025-04-01 14:30:00] G1: Implement authentication system
  └── Merged: S2 at 2025-04-01 14:30:00

[2025-04-01 14:30:22] G1: Implement authentication system
  └── Completed: 2025-04-01 14:30:22

Summary: 5/9 goals completed
```

## Graphical Visualizations

### Graph Visualization (`goal graph`)

The `goal graph` command generates a graphical representation of your goal hierarchy using Graphviz. The output is saved as both PDF and PNG files in the `.goal/visualization` directory.

Example graph output:

```
digraph Goals {
  rankdir=TB;
  node [shape=box, style=filled, fontname=Arial];
  edge [fontname=Arial];
  
  "G1" [label="G1\nImplement authentication system", fillcolor=green];
  "S1" [label="S1\nImplement user registration", fillcolor=green];
  "S4" [label="S4\nDesign registration form", fillcolor=gray];
  "S2" [label="S2\nImplement login/logout", fillcolor=orange];
  "S5" [label="S5\nCreate login form", fillcolor=green];
  "S6" [label="S6\nImplement session management", fillcolor=green];
  "S3" [label="S3\nImplement password reset", fillcolor=lightblue];
  "S7" [label="S7\nDesign password reset flow", fillcolor=gray];
  
  "G1" -> "S1";
  "G1" -> "S2";
  "G1" -> "S3";
  "S1" -> "S4";
  "S2" -> "S5";
  "S2" -> "S6";
  "S3" -> "S7";
}
```

When rendered, this graph shows:
- Nodes representing goals and subgoals
- Node colors indicating status:
  - Green: Complete goals
  - Orange: Goals with all subgoals complete (ready to merge)
  - Light blue: Incomplete goals with subgoals
  - Gray: Goals with no subgoals

### Example Graph Rendering

```
                ┌───────────────────────┐
                │          G1           │
                │ Implement             │
                │ authentication system │
                └───────────┬───────────┘
                            │
          ┌────────────┬────┼────────────┐
          │            │                 │
┌─────────▼────────┐ ┌─▼──────────────┐ ┌▼───────────────┐
│       S1         │ │      S2        │ │     S3         │
│ Implement user   │ │ Implement      │ │ Implement      │
│ registration     │ │ login/logout   │ │ password reset │
└────────┬─────────┘ └───────┬────────┘ └───────┬────────┘
         │                   │                  │
┌────────▼─────────┐    ┌────┴───────┐   ┌─────▼────────┐
│      S4          │    │            │   │     S7       │
│ Design           │ ┌──▼───────┐ ┌──▼───┐ Design       │
│ registration form│ │   S5     │ │  S6    │ password    │
└──────────────────┘ │ Create   │ │Implement│ reset flow │
                     │ login    │ │session │ └────────────┘
                     │ form     │ │mgmt    │
                     └──────────┘ └────────┘
```

## Using Visualization for Project Management

The visualization tools in the goal management CLI can help with:

1. **Progress Tracking**: Quickly assess which goals are complete and which need attention
2. **Team Communication**: Share the goal structure with team members
3. **Planning**: Identify which subgoals are ready to be merged into parent goals
4. **Documentation**: Generate visual documentation of project progress
5. **Reporting**: Create reports showing goal completion over time

For best results:
- Use `goal tree` for daily progress tracking
- Use `goal status` for detailed status reviews
- Use `goal history` to review the chronological progress
- Use `goal graph` for creating documentation and reports 