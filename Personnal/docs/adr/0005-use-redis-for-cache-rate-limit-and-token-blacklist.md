# 0005 使用 Redis 支撑缓存、限流和 Token 黑名单

## 状态

Accepted

## 背景

项目需要处理几类短生命周期状态：

- 用户详情缓存。
- JWT 登出后的黑名单。
- API 限流计数。

这些数据不适合全部放 MySQL，因为访问频繁、生命周期短、需要过期时间。

## 决策

使用 Redis：

- Django cache 保存用户信息和 token 黑名单。
- FastAPI Redis 保存限流计数。
- FastAPI 查询 Redis 判断 token 是否被拉黑。
- FastAPI 查询或写入用户信息缓存。

## 备选方案

- 全部使用 MySQL。
- 内存缓存。
- API 网关限流。
- 专门认证服务维护 token 状态。

## 后果

正面影响：

- Redis 支持 TTL，适合缓存和黑名单。
- 限流实现简单。
- 减少用户详情接口重复查询。
- 两个后端都能共享 token 失效状态。

负面影响：

- Redis 成为关键依赖。
- 两个服务的 Redis DB、key 前缀和序列化方式需要统一。
- 本地开发需要额外启动 Redis。

## 后续约束

- FastAPI Redis 配置建议改为环境变量。
- Django 和 FastAPI 需要明确 key 约定。
- 生产环境 Redis 需要配置密码、持久化和监控。

