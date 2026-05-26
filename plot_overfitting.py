import re
import matplotlib.pyplot as plt


# =========================
# 修改这里的路径
# =========================
missrec_log_path = '/mnt/data/zyj/MM23-MISSRec/log/MISSRec/May-20-2026_13-47-32.log'
ours_log_path = '/mnt/data/zyj/MM23-MISSRec/log/MISSRec/May-20-2026_16-27-05.log'


# =========================
# log解析函数
# =========================
def parse_log(log_path):
    epochs = []
    train_losses = []
    hit10s = []
    ndcg10s = []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = None
    current_loss = None

    for i, line in enumerate(lines):

        # -------------------------
        # 提取 epoch + train loss
        # -------------------------
        train_match = re.search(
            r'epoch\s+(\d+)\s+training.*train loss:\s*([0-9.]+)',
            line
        )

        if train_match:
            current_epoch = int(train_match.group(1))
            current_loss = float(train_match.group(2))
            continue


        # -------------------------
        # 提取 valid result
        # -------------------------
        if 'valid result' in line:

            # 下一行通常包含指标
            if i + 1 < len(lines):
                metric_line = lines[i + 1]

                hit_match = re.search(r'hit@10\s*:\s*([0-9.]+)', metric_line)
                ndcg_match = re.search(r'ndcg@10\s*:\s*([0-9.]+)', metric_line)

                if current_epoch is not None and current_loss is not None \
                        and hit_match and ndcg_match:

                    epochs.append(current_epoch)
                    train_losses.append(current_loss)
                    hit10s.append(float(hit_match.group(1)))
                    ndcg10s.append(float(ndcg_match.group(1)))

    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'hit10': hit10s,
        'ndcg10': ndcg10s
    }


# =========================
# 读取日志
# =========================
missrec = parse_log(missrec_log_path)
ours = parse_log(ours_log_path)


# =========================
# 统一论文风格
# =========================
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (6, 4)


# ======================================================
# 图1 Train Loss
# ======================================================
plt.figure()

plt.plot(
    missrec['epochs'],
    missrec['train_loss'],
    label='MISSRec'
)

plt.plot(
    ours['epochs'],
    ours['train_loss'],
    label='Ours'
)

plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Analysis')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/mnt/data/zyj/MM23-MISSRec/picture/pantry/train_loss_curve.png', dpi=300)


# ======================================================
# 图2 HIT@10
# ======================================================
plt.figure()

plt.plot(
    missrec['epochs'],
    missrec['hit10'],
    label='MISSRec'
)

plt.plot(
    ours['epochs'],
    ours['hit10'],
    label='Ours'
)

plt.xlabel('Epoch')
plt.ylabel('Valid HIT@10')
plt.title('Validation HIT@10 Analysis')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/mnt/data/zyj/MM23-MISSRec/picture/pantry/valid_hit10_curve.png', dpi=300)


# ======================================================
# 图3 NDCG@10
# ======================================================
plt.figure()

plt.plot(
    missrec['epochs'],
    missrec['ndcg10'],
    label='MISSRec'
)

plt.plot(
    ours['epochs'],
    ours['ndcg10'],
    label='Ours'
)

plt.xlabel('Epoch')
plt.ylabel('Valid NDCG@10')
plt.title('Validation NDCG@10 Analysis')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/mnt/data/zyj/MM23-MISSRec/picture/pantry/valid_ndcg10_curve.png', dpi=300)


# =========================
# 显示图像
# =========================
plt.show()


print('Done!')
print('Generated:')
print('- train_loss_curve.png')
print('- valid_hit10_curve.png')
print('- valid_ndcg10_curve.png')


