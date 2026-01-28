首先是一些环境的创建，

主要是

```
ActionsCfg
```


```
ObservationsCfg
```

以及事件


```
EventCfg
```


# 强化学习环境


```
RewardsCfg
```


需要加入回合，目的是失败了可以重新开始，或者说已经保持稳定的话想要换一个初始姿势进行训练

因此就要设置回合终止条件


```
TerminationsCfg
```


还有就是多阶段的课程的定义课程
