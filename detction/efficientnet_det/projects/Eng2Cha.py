# 为了显示可理解的标签，又省去麻烦的配置
# 我们用google翻译来解决一切吧

a = ['一次性快餐盒', '书籍纸张', '充电宝', '剩饭剩菜', '包', '垃圾桶', '塑料器皿', '塑料玩具', '塑料衣架', '大骨头', '干电池', '快递纸袋', '插头电线', '旧衣服', '易拉罐', '枕头', '果皮果肉', '毛绒玩具', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '牙签', '玻璃器皿', '砧板', '筷子', '纸盒纸箱', '花盆', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '软膏', '过期药物', '酒瓶', '金属厨具', '金属器皿', '金属食品罐', '锅', '陶瓷器皿', '鞋', '食用油桶', '饮料瓶', '鱼骨']
b = ['Disposable fast food box', 'Book paper', 'Charge treasure', 'Leftovers',' Package ',' Trash can ',' Plastic utensils', 'Plastic toys',' Plastic hangers', ' Big bones, 'Dry batteries',' Express paper bags', 'Plug and wires',' Old clothes', 'Easy cans',' Pillows', 'Peel flesh', 'Stuffed toys',' Stained plastic ',' Stained Damaged paper ',' Hygiene ',' Cigarette ',' Toothpick ',' Glassware ',' Cutting board ',' Chopsticks ',' Carton box ',' Flower pot ',' Tea residue ',' Dish Help Vegetable Leaf ',' Eggshell ',' Seasoning Bottle ',' Ointment ',' Expired Medicine ',' Wine Bottle ',' Metal Kitchenware ',' Metal Utensil ',' Metal Food Jar ',' Pan ',' Ceramic ware ',' shoes', 'edible oil barrels',' drink bottles', 'fish bones']

c = [b[i]:a[i] for i in len(a)]

print(c)