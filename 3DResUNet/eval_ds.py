import SimpleITK as sitk
import numpy as np
import torch
from ResUNet import ResUNet
import parameter as para
import metrics

def liverSegment(src_ct_image):
    net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
    net.load_state_dict(torch.load('./weights/weights.pth'))
    net.eval()

    ct = sitk.Cast(src_ct_image, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200
    # too_small = False
    # if ct_array.shape[0] < para.size:
    #     depth = ct_array.shape[0]
    #     temp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale))) * para.lower
    #     temp[0: depth] = ct_array
    #     ct_array = temp 
    #     too_small = True

    # ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3)

    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], 256, 256), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 256, 256), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1])
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs      
            
            start_slice += para.stride
            end_slice = start_slice + para.size - 1

        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1])
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs
        
        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold * count)] = 1

        # if too_small:
        #     temp = np.zeros((depth, 256, 256), dtype=np.float32)
        #     temp += pred_seg[0: depth]
        #     pred_seg = temp

        # pred_seg_down = ndimage.zoom(pred_seg, (1, 1/para.down_scale, 1/para.down_scale), order=0)

        pred_seg_im = sitk.GetImageFromArray(pred_seg)
        pred_seg_im.SetDirection(ct.GetDirection())
        pred_seg_im.SetOrigin(ct.GetOrigin())
        pred_seg_im.SetSpacing(ct.GetSpacing())
        pred_seg_im = sitk.Cast(pred_seg_im, sitk.sitkUInt8)

        # Remove non connected component
        cc = sitk.ConnectedComponent(pred_seg_im,True)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(cc,pred_seg_im)
        largestCClabel = 0
        largestCCsize = 0 

        for l in stats.GetLabels():
            if int(stats.GetPhysicalSize(l)) >= largestCCsize:
                largestCCsize = int(stats.GetPhysicalSize(l))
                largestCClabel = l

        img_seg_sitk = cc == largestCClabel # get the largest component
        img_seg_out = sitk.GetArrayFromImage(img_seg_sitk)   

        return img_seg_out, img_seg_sitk
        # return pred_seg, pred_seg_im
        
# Load ct and label
ct_sitk = sitk.ReadImage('./data/ct.nii.gz')
label_sitk = sitk.ReadImage('./data/label.nii.gz')
label = sitk.GetArrayFromImage(label_sitk)

# Infer
pred, pred_sitk = liverSegment(ct_sitk)
sitk.WriteImage(pred_sitk, './data/seg.nii.gz')

# Evaluate
Loss = metrics.Metric(label, pred, (1,1,1))

DSC = Loss.get_dice_coefficient()
IoU = Loss.get_jaccard_index()
VOE = Loss.get_VOE()
FNR = Loss.get_FNR()
FPR = Loss.get_FPR()
ASSD = Loss.get_ASSD()
RMSD = Loss.get_RMSD()
MSD = Loss.get_MSD()

print(DSC)
