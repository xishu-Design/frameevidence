# Original Chinese inputs and English translations

S identifiers below are trace-only labels, not additions to the delivered prompts.

## group_00392

以图一镜头朝向为正前方，面对前方时右手侧为右方，烤箱位于椅子的哪一侧？

Take the camera's viewing direction in Image 1 as forward, and the right-hand side when facing forward as right. On which side of the chair is the oven?

- A. 前方 / Front
- B. 右方 / Right
- C. 后方 / Behind
- D. 信息不足 / Insufficient information

The released deletion removes S01. The first image delivered to the model is then S02, while the unchanged question still refers to Image 1.

### one_valid_mss

Image order: S01 → S02 → S03 → S04. Released answer reference: B.

```text
请根据当前提供的信息回答下面的四选一问题。
以图一镜头朝向为正前方，面对前方时右手侧为右方，烤箱位于椅子的哪一侧？
A. 前方
B. 右方
C. 后方
D. 信息不足
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### remove_required_view

Image order: S02 → S03 → S04. Released answer reference: D.

```text
请根据当前提供的信息回答下面的四选一问题。
以图一镜头朝向为正前方，面对前方时右手侧为右方，烤箱位于椅子的哪一侧？
A. 前方
B. 右方
C. 后方
D. 信息不足
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### hard_distractor_added

Image order: S01 → S02 → S03 → S04 → S05. Released answer reference: B.

```text
请根据当前提供的信息回答下面的四选一问题。
以图一镜头朝向为正前方，面对前方时右手侧为右方，烤箱位于椅子的哪一侧？
A. 前方
B. 右方
C. 后方
D. 信息不足
只输出一个答案字母（A、B、C 或 D），不要解释。
```

## group_00368

以图二镜头朝向为正前方，面对前方时右手侧为右方，盆栽位于桌子的哪一侧？

Take the camera's viewing direction in Image 2 as forward, and the right-hand side when facing forward as right. On which side of the table is the potted plant?

- A. 右后方 / Back-right
- B. 右前方 / Front-right
- C. 信息不足 / Insufficient information
- D. 左后方 / Back-left

The released deletion removes S01 and retains the original anchor S02 in the first position. The second image delivered to the model is now S03, while the question still refers to Image 2.

### one_valid_mss

Image order: S01 → S02 → S03. Released answer reference: A.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方，盆栽位于桌子的哪一侧？
A. 右后方
B. 右前方
C. 信息不足
D. 左后方
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### remove_required_view

Image order: S02 → S03. Released answer reference: C.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方，盆栽位于桌子的哪一侧？
A. 右后方
B. 右前方
C. 信息不足
D. 左后方
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### hard_distractor_added

Image order: S01 → S02 → S03 → S04. Released answer reference: A.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方，盆栽位于桌子的哪一侧？
A. 右后方
B. 右前方
C. 信息不足
D. 左后方
只输出一个答案字母（A、B、C 或 D），不要解释。
```

## group_00311

以图二镜头朝向为正前方，面对前方时右手侧为右方,游戏手柄位于矿泉水瓶的那一侧？

Take the camera's viewing direction in Image 2 as forward, and the right-hand side when facing forward as right. On which side of the bottled water is the game controller?

- A. 左后方 / Back-left
- B. 信息不足 / Insufficient information
- C. 右后方 / Back-right
- D. 左前方 / Front-left

The released deletion removes S03. The original anchor S02 remains in the second position, so the explicit ordinal continues to identify the same image.

### one_valid_mss

Image order: S01 → S02 → S03. Released answer reference: D.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方,游戏手柄位于矿泉水瓶的那一侧？
A. 左后方
B. 信息不足
C. 右后方
D. 左前方
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### remove_required_view

Image order: S01 → S02. Released answer reference: B.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方,游戏手柄位于矿泉水瓶的那一侧？
A. 左后方
B. 信息不足
C. 右后方
D. 左前方
只输出一个答案字母（A、B、C 或 D），不要解释。
```

### hard_distractor_added

Image order: S01 → S02 → S03 → S04. Released answer reference: D.

```text
请根据当前提供的信息回答下面的四选一问题。
以图二镜头朝向为正前方，面对前方时右手侧为右方,游戏手柄位于矿泉水瓶的那一侧？
A. 左后方
B. 信息不足
C. 右后方
D. 左前方
只输出一个答案字母（A、B、C 或 D），不要解释。
```
