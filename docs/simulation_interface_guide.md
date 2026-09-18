# 仿真接口说明表

仿真接口实现：`TongSimGrpcClient`  
默认地址：`127.0.0.1:50060`  
配置字段：`tongsim_server_endpoint`

## 912 赛题系统实测接口

对运行中的 912 服务端逐个调用 43 个候选方法得到的结果。**proto 声明不可信**：仓库里
的 `tongsim_service.proto` 与 912 双向不一致——它声明了 912 没有的 `get_object_in_hand`，
又漏掉了 912 有的 `has_object_in_hand`、`put_down_sth`、`move_to_npc`、`open_door`、
`close_door`、`interact`。判断接口是否存在只能实际调用一次，看是否返回 `UNIMPLEMENTED`。

复现：`uv run python scripts/probe_912_rpcs.py`（需 UE 客户端与 tongsim_server 同时在跑）

### 912 提供

| 分类 | 接口 |
|---|---|
| 感知 | `acquire_first_person_perception` |
| 查看 | `look_at_location`、`look_at_object`、`point_at_object` |
| 移动 | `move_to_location`、`move_forward`、`move_to_object`、`move_to_npc` |
| 抓取 | `move_and_take_object`、`transfer_puzzle_piece`、`set_pickup_whitelist` |
| 放置 | `put_down_sth`、`move_and_put_down`、`move_and_put_down_object_in_container` |
| 手持 | `has_object_in_hand` |
| 场景交互 | `pour_water`、`slice_food`、`wash_hands`、`wash_object_in_hand`、`sit_down_to_object`、`mop_floor`、`rest`、`open_door`、`close_door`、`interact` |
| 对话 | `speak_to_npc` |
| 生命周期 | `spawn_character`、`destory_character`、`close`、`heartbeat` |

### 912 已删除

`acquire_first_person_image`、`acquire_first_person_segmantic_image`、
`fetch_first_person_visible_objects`、`get_object_basic_info`、`get_object_world_aabb`、
`get_object_id_by_name`、`get_object_in_hand`、`put_down_to_location`、`set_object_pose`、
`move_and_take_puzzle_piece`。

### 统一感知的响应结构

`acquire_first_person_perception(character_id, width, height)` 返回：

- `image`：**服务端已经拼好的合成图**，左半 RGB、右半带数字标签的分割图，数字即 `object_id`。
  对齐与标注都在服务端完成，客户端不再自己拼图。
- `objects`：可见物体列表，每项含 `object_id`、`shape`、`color`、`place_location`、
  `rotation`、`world_aabb`，**但没有 `segmentation_id`**。

两个后果：旧的五个感知接口全部折叠进这一个调用；画面与物体列表由服务端保证同帧，
本地无法再计算像素覆盖率，`last_perception_diagnostics` 留空，依赖它的两个一致性检查
会以 `diagnostics_unavailable` 放行。

## 新旧服务端兼容策略

`TongSimGrpcClient` 对两版服务端都可用：

- 新接口用 `channel.unary_unary(完整方法路径)` 直接拨号，绕开过期的 proto；只有旧版
  才有的方法反过来沿用生成的 stub。
- 服务端返回 `UNIMPLEMENTED` 时客户端返回 `None`（`acquire_first_person_perception`、
  `has_object_in_hand`、`move_to_npc`），调用方据此回退到旧接口。`SemanticMapper` 的
  两条路径分别是 `_perception_from_unified` 与 `_perception_from_split`。
- 912 没有按物体查询 AABB 的接口，`get_object_world_aabb` 改用最近一次统一感知的缓存；
  物体不在视野内时返回 `{}`，调用方保留此前缓存的包围盒。
- 912 的 `has_object_in_hand` 只回答"手里有没有东西"，不回答是什么。抓取是获得物体的
  唯一途径，因此 `VLMAgent` 记住最近一次抓取目标来还原手中物体 ID。

## 角色生命周期

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `spawn_character` | `asset_name`, `loc`, `rot`, `desired_name`, `fov`, `width`, `height` | `character_id` | 在仿真中生成角色 |
| `destory_character` | `character_id` | `dict` | 销毁角色 |
| `heartbeat` | - | `dict` | 保持 TongSim 连接活跃 |
| `close` | - | - | 关闭 TongSim 连接并清理资源 |

## 感知查询

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `acquire_first_person_image` | `character_id` | `image` | 获取第一视角 RGB 图像，通常为 base64 |
| `acquire_first_person_segmantic_image` | `character_id` | `image` | 获取第一视角语义分割图 |
| `fetch_first_person_visible_objects` | `character_id` | `objects` | 获取第一视角可见物体列表 |
| `get_object_basic_info` | `object_id` | `dict` | 获取物体颜色、形状、位置等基础信息 |
| `get_object_world_aabb` | `object_id` | `dict` | 获取物体世界坐标 AABB 包围盒 |
| `get_object_id_by_name` | `name` | `object_id` 或 `None` | 根据物体名查找 object id |
| `get_object_in_hand` | `character_id` | `(object_id, hand_idx)` 或 `None` | 查询角色当前手中物体 |

## 视角控制

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `look_at_location` | `character_id`, `target_location`, `is_cancel`, `execute_immediately` | `dict` | 看向指定坐标 |
| `look_at_object` | `character_id`, `object_id`, `is_cancel` | `dict` | 看向指定物体 |
| `point_at_object` | `character_id`, `object_id`, `is_cancel`, `which_hand` | `dict` | 指向指定物体 |

## 移动与抓取

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `move_to_location` | `character_id`, `target_location`, `stop_distance` | `dict` | 移动到指定坐标 |
| `move_forward` | `character_id`, `distance` | `dict` | 向前移动指定距离 |
| `move_to_object` | `character_id`, `object_id` | `dict` | 移动到指定物体附近 |
| `move_and_take_object` | `character_id`, `object_id`, `which_hand` | `dict` | 移动到物体并抓取 |
| `turn_in_degree` | `character_id`, `degree` | `dict` | 原地旋转指定角度 |

## 放置与物体操作

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `put_down_to_location` | `character_id`, `target_location`, `which_hand`, `disable_physics`, `hold_if_unreachable`, `force_release`, `auto_rotate`, `rotation`, `force_locate` | `dict` | 将手中物体放到指定位置 |
| `move_and_put_down` | `character_id`, `move_target_location`, `put_target_location`, `which_hand`, `put_rotation` | `dict` | 移动到指定位置后放下手中物体 |
| `move_and_put_down_object_in_container` | `character_id`, `which_hand` | `dict` | 将手中物体放入容器 |
| `set_object_pose` | `object_id`, `location`, `rotation` | `bool` | 直接设置物体位置和旋转 |
| `pour_water` | `character_id`, `object_id`, `location`, `which_hand` | `dict` | 倒水 |
| `slice_food` | `character_id`, `object_id`, `location` | `dict` | 切食物 |
| `wash_hands` | `character_id`, `faucet_object_id` | `dict` | 洗手 |
| `wash_object_in_hand` | `character_id`, `faucet_object_id` | `dict` | 清洗手中物体 |

## 场景交互

| 接口 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `open_door` | `character_id`, `door_id`, `which_hand` | `dict` | 开门 |
| `close_door` | `character_id`, `door_id`, `which_hand` | `dict` | 关门 |
| `sit_down_to_object` | `character_id`, `object_id` | `dict` | 坐到指定物体 |
| `mop_floor` | `character_id`, `dirt_id` | `dict` | 拖地 |
| `rest` | `character_id` | `dict` | 休息 |
| `speak_to_npc` | `character_id`, `target`, `content` | `dict` | 与 NPC 对话 |
| `interact` | `character_id`, `object_id`, `new_object_state` | `dict` | 切换物体状态 |

## 常用参数格式

| 参数 | 格式 | 示例 |
|---|---|---|
| `target_location` / `location` | 三维坐标列表或字典 | `[100.0, 200.0, 50.0]` |
| `rotation` / `put_rotation` | `roll`, `yaw`, `pitch` | `{"roll": 0, "yaw": 90, "pitch": 0}` |
| `which_hand` | 整数 | `0` |
| `object_id` | TongSim 原始物体 ID | `"BP_Cup_12"` |
