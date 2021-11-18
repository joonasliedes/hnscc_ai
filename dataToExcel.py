import xlsxwriter


def loadToExcel(filename, iou_scores, dice_scores): 
# Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

   
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0
    worksheet.write(row,col,"Iter")
    worksheet.write(row,col+1,"IoU")
    worksheet.write(row,col+2,"Dice")

    row = 1
    # Iterate over the data and write it out row by row.
    for iou, dice in zip(iou_scores, dice_scores):
        worksheet.write(row, col, str(row))
        worksheet.write(row, col + 1, iou)
        worksheet.write(row, col + 2, dice)
        row += 1

    workbook.close()