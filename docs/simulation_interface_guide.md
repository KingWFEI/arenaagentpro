# 仿真接口说明表

仿真接口实现：`TongSimGrpcClient`  
默认地址：`127.0.0.1:50060`  
配置字段：`tongsim_server_endpoint`

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
