---
type: weekly-review
week: <% tp.file.title %>
date: <% tp.date.now("YYYY-MM-DD") %>
tags:
  - weekly-review
---
<%*
const parsedWeek = moment(tp.file.title, "gggg-[W]ww", true);
const activeWeek = parsedWeek.isValid() ? parsedWeek : moment();
const weekStart = activeWeek.clone().startOf("week").format("YYYY-MM-DD");
const weekEnd = activeWeek.clone().endOf("week").format("YYYY-MM-DD");
const queryStart = activeWeek.clone().startOf("week").subtract(1, "day").format("YYYY-MM-DD");
const queryEnd = activeWeek.clone().endOf("week").add(1, "day").format("YYYY-MM-DD");
-%>
# <% tp.file.title %> 周复盘

> 本周范围：<% weekStart %> ～ <% weekEnd %>

## 本周完成

```tasks
done
done after <% queryStart %>
done before <% queryEnd %>
```

## 未完成与延续

```tasks
not done
happens this week
```

## 本周亮点

- 

## 遇到的问题

- 

## 下周重点

- [ ] 

## 复盘与调整

- 
