import codecs
# 创建一个空字典来存储映射
goods_dict = {}
koods_dict = {}
added_dict = {}



with open('Alist.txt', 'r') as f:
    for line in f:
        # 假设每行的结构是 '序号\t商品名称\n'
        number1, name1, _  = line.strip().split('\t')  # 使用适当的分隔符替换'\t'
        koods_dict[name1] = number1

# 打开和读取 Blist.txt 文件
with open('Blist.txt', 'r') as f:
    for line in f:
        # 假设每行的结构是 '序号\t商品名称\n'
        number, name, _  = line.strip().split('\t')  # 使用适当的分隔符替换'\t'
        goods_dict[name] = number

# 打开你的商品名称文件并创建一个新的输出文件
with open('train_data.txt', 'r') as f, open('traindata_new.txt', 'w') as out_f:
    for line in f:
        names = line.strip().split('\t')
        for name in names:
            if name in goods_dict:
                print(f"name: {name}")
                print(f"goods_dict[name] before addition: {goods_dict[name]}")
                if name not in added_dict:  # Only add 14636 if name is not in added_dict
                    goods_dict[name] = str(int(goods_dict[name]) + 14636)
                    added_dict[name] = True  # Mark the name as added
                print(f"goods_dict[name] after addition: {goods_dict[name]}")
                out_f.write(goods_dict[name] + '\t')
            elif name in koods_dict:
                out_f.write(koods_dict[name] + '\t')
            if name not in goods_dict and name not in koods_dict:
                print(f'Warning: 商品名称 "{name}" 在 AB_dict.txt 文件中不存在.')
        out_f.write('\n')