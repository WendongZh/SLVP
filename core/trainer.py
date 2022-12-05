import os.path
import datetime
from unicodedata import category
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from core.utils import preprocess
import lpips

def train(model, ims, real_input_flag, configs, itr,category):
    cost, cost_pd, cost_kl = model.train(ims, real_input_flag,category)

    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost_, cost_pd_, cost_kl_ = model.train(ims_rev,real_input_flag,category)
        cost = (cost + cost_) / 2
        cost_pd = (cost_pd + cost_pd_) / 2
        cost_kl = (cost_kl + cost_kl_) / 2
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
        print('loss: ' + str(cost) + ', pd: ' + str(cost_pd) + ', kl: ' + str(cost_kl))
    return cost

def CPL_train(model,pre_model, ims, real_input_flag, configs, itr,category, is_replay=False):
    
    cost, cost_pd, cost_kl,cost_cat,loss_recon = model.CPL_train(pre_model,ims, real_input_flag,category,itr,is_replay)

    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost_, cost_pd_, cost_kl_,loss_recon= model.CPL_train(pre_model,ims_rev,real_input_flag,category,itr,is_replay)
        cost = (cost + cost_) / 2
        cost_pd = (cost_pd + cost_pd_) / 2
        cost_kl = (cost_kl + cost_kl_) / 2
        cost_cat = (cost_cat + cost_cat) / 2
    
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
        print('loss: ' + str(cost) + ', pd: ' + str(cost_pd) +',recon'+str(loss_recon)+ ', kl: ' + str(cost_kl)+ ', cat: ' + str(cost_cat))
    return cost

def test(model, test_model,test_input_handle, configs, itr,category=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs.gen_frm_dir, 'itr_'+str(itr))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    test_input_handle.begin(do_shuffle=False)

    batch_id = 0
    best_mse, best_ssim, best_psnr, best_mae, best_sharp = [], [], [], [], []
    avg_mse, avg_ssim, avg_psnr, avg_mae, avg_sharp = [], [], [], [], []
    worst_mse, worst_ssim, worst_psnr, worst_mae, worst_sharp = [], [], [], [], []

    best_last_mse,worst_last_mse=[],[]
    img_mse, ssim = [], []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
    allright_cat_num=0
    while (test_input_handle.no_batch_left() == False ):
        sample=np.zeros(
        (configs.batch_size* configs.num_samples,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))
        

        test_ims ,test_cat, _= test_input_handle.get_batch()

        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        img_gen,right_cat_num = model.test(test_model,test_dat, sample,category)
        allright_cat_num+=right_cat_num

        #img_gen = np.concatenate(img_gen)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        gt = test_ims[:, -output_length:]
        gx = img_gen[:, -output_length:]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)

        best_mse_id, worst_mse_id, best_ssim_id, worst_ssim_id = [], [], [], []
        for bid in range(configs.batch_size):
            seq_id = batch_id * configs.batch_size + bid
            for sid in range(configs.num_samples):
                sid_in_batch = sid*configs.batch_size + bid
                mse_sample = np.square(gt[bid] - gx[sid_in_batch]).sum()
                #print(mse_sample)
                ssim_sample, psnr_sample, mae_sample, sharp_sample = 0, 0, 0, 0
                for i in range(output_length):
                    gt_frm = gt[bid, i]
                    gx_frm = gx[sid*configs.batch_size + bid, i]

                    img_mse[i] += np.square(gt[bid,i] - gx[sid_in_batch,i]).sum()
                    gt_frm = np.uint8(gt_frm * 255)
                    gx_frm = np.uint8(gx_frm * 255)

                    score, _ = compare_ssim(gx_frm, gt_frm, full=True, multichannel=True)
                    ssim_sample += score
                    ssim[i]+=score

                '''[b1s1 b2s1 b3s1, b1s2 b2s2 b3s2]'''
                if sid == 0:
                    best_mse.append(mse_sample)
                    avg_mse.append(mse_sample)
                    worst_mse.append(mse_sample)
                    best_mse_id.append(sid_in_batch)
                    worst_mse_id.append(sid_in_batch)
                    
                    best_last_mse.append(img_mse[output_length-1])
                    worst_last_mse.append(img_mse[output_length-1])
                    
                    best_ssim.append(ssim_sample)
                    avg_ssim.append(ssim_sample)
                    worst_ssim.append(ssim_sample)
                    best_ssim_id.append(sid_in_batch)
                    worst_ssim_id.append(sid_in_batch)

                else:
                    if best_mse[-1] > mse_sample:
                        best_mse[-1] = mse_sample
                        best_mse_id[-1] = sid_in_batch
                    if worst_mse[-1] < mse_sample:
                        worst_mse[-1] = mse_sample
                        worst_mse_id[-1] = sid_in_batch
                    if worst_last_mse[-1]<img_mse[output_length-1]:
                        worst_last_mse[-1]=img_mse[output_length-1]
                    avg_mse[seq_id] += mse_sample

                    if best_ssim[-1] < ssim_sample:
                        best_ssim[-1] = ssim_sample
                        best_ssim_id[-1] = sid_in_batch
                    if worst_ssim[-1] > ssim_sample:
                        worst_ssim[-1] = ssim_sample
                        worst_ssim_id[-1] = sid_in_batch
                    if best_last_mse[-1]>img_mse[output_length-1]:
                        best_last_mse[-1]=img_mse[output_length-1]
                    avg_ssim[seq_id] += ssim_sample


            #save prediction examples
            
            if batch_id == 0:
                if batch_id == 0:
                    batch_path = os.path.join(res_path, str(bid))
                    if not os.path.exists(batch_path):
                        os.mkdir(batch_path)
                    best_mse_path = os.path.join(batch_path, 'best_mse')
                    if not os.path.exists(best_mse_path):
                        os.mkdir(best_mse_path)
                    worst_mse_path = os.path.join(batch_path, 'worst_mse')
                    if not os.path.exists(worst_mse_path):
                        os.mkdir(worst_mse_path)
                    best_ssim_path = os.path.join(batch_path, 'best_ssim')
                    if not os.path.exists(best_ssim_path):
                        os.mkdir(best_ssim_path)
                    worst_ssim_path = os.path.join(batch_path, 'worst_ssim')
                    if not os.path.exists(worst_ssim_path):
                        os.mkdir(worst_ssim_path)

                    for i in range(configs.total_length):
                        if i < 9:
                            name = 'gt0' + str(i + 1) + '.png'
                        else:
                            name = 'gt' + str(i + 1) + '.png'
                        file_name = os.path.join(best_mse_path, name)
                        img_gt = np.uint8(test_ims[bid, i] * 255)
                        cv2.imwrite(file_name, img_gt)
                    for i in range(img_gen_length):
                        if i < 8:
                            name = 'pd0' + str(i + 2) + '.png'
                        else:
                            name = 'pd' + str(i + 2) + '.png'
                        file_name = os.path.join(best_mse_path, name)
                        img_pd = img_gen[best_mse_id[bid], i]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)

                        file_name = os.path.join(worst_mse_path, name)
                        img_pd = img_gen[worst_mse_id[bid], i]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)

                        file_name = os.path.join(best_ssim_path, name)
                        img_pd = img_gen[best_ssim_id[bid], i]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)

                        file_name = os.path.join(worst_ssim_path, name)
                        img_pd = img_gen[worst_ssim_id[bid], i]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)
            

        batch_id = batch_id + 1  # batch_size=1 for test
        test_input_handle.next()
    # Average BEST and WORST predictions over all seqs
    # print(len(avg_mse))
    # print(batch_id * configs.batch_size*configs.num_samples)
    best_mse = np.mean(np.asarray(best_mse, dtype=np.float32)) / output_length
    avg_mse = np.mean(np.asarray(avg_mse, dtype=np.float32))/configs.num_samples
    worst_mse = np.mean(np.asarray(worst_mse, dtype=np.float32)) / output_length

    best_last_mse=np.mean(np.asarray(best_last_mse, dtype=np.float32))
    worst_last_mse=np.mean(np.asarray(worst_last_mse, dtype=np.float32))
    # print('best mse last frame: ' + str(best_last_mse))
    # print('worst mse last frame: ' + str(worst_last_mse))

    best_ssim = np.mean(np.asarray(best_ssim, dtype=np.float32)) / output_length
    avg_ssim = np.mean(np.asarray(avg_ssim, dtype=np.float32))/ (output_length*configs.num_samples)
    worst_ssim = np.mean(np.asarray(worst_ssim, dtype=np.float32)) / output_length

    print('best mse per frame: ' + str(best_mse))
    print('avg mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size*configs.num_samples))
    print('worst mse per frame: ' + str(worst_mse))

    print('best ssim per frame: ' + str(best_ssim))
    print('avg ssim per frame: ' + str(avg_ssim))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i]/ (batch_id * configs.batch_size*configs.num_samples))
    print('worst ssim per frame: ' + str(worst_ssim))

    cluster_accuracy=allright_cat_num/(configs.batch_size * batch_id*configs.num_samples)
    # print('cluster_accuracy'+str(cluster_accuracy))
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id*configs.num_samples)
    record = dict()
    record["avg_mse_per_seq"] = avg_mse
    record["img_mse_per_frame"] = []
    for i in range(configs.total_length - configs.input_length):
        record["img_mse_per_frame"].append(img_mse[i] / (batch_id * configs.batch_size*configs.num_samples))
    record["avg_ssim_per_seq"] = avg_ssim
    record["img_ssim_per_frame"] = []
    for i in range(configs.total_length - configs.input_length):
        record["img_ssim_per_frame"].append(ssim[i] / (batch_id * configs.batch_size*configs.num_samples))
    return record


def test_final_lpips(model, test_model,test_input_handle, configs, itr,category=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs.gen_frm_dir, 'itr_'+str(itr))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    test_input_handle.begin(do_shuffle=False)

    batch_id = 0
    best_mse, best_ssim, best_psnr, best_mae, best_sharp = [], [], [], [], []
    avg_mse, avg_ssim, avg_psnr, avg_mae, avg_sharp = [], [], [], [], []
    worst_mse, worst_ssim, worst_psnr, worst_mae, worst_sharp = [], [], [], [], []

    # best_last_mse,worst_last_mse=[],[]
    best_last_psnr,worst_last_psnr=[],[]
    img_mse, ssim, img_psnr, img_lpips = [], [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        img_psnr.append(0)
        img_lpips.append(0)

    allright_cat_num=0

    batch_size=configs.batch_size
    avg_psnr_all = []
    avg_ssim_all = []
    avg_lpips_all = []

    mean_psnr_all = []

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.to(configs.device)

    while (test_input_handle.no_batch_left() == False ):

        sample=np.zeros(
        (batch_size* configs.num_samples,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel)) 
        

        test_ims ,test_cat, _= test_input_handle.get_batch()

        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        if configs.model_name=='CPL_base':
            img_gen, right_cat_num= model.test(test_model,test_dat, sample,test_cat)
            allright_cat_num+=right_cat_num
        elif configs.model_name=='CPL':
            img_gen,right_cat_num, img_gt_lpips, img_gen_lpips = model.test_final_lpips(test_model,test_dat, sample, category)
            allright_cat_num+=right_cat_num
        #img_gen = np.concatenate(img_gen)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        gt = test_ims[:, -output_length:]
        gx = img_gen[:, -output_length:]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)

        # best_mse_id, worst_mse_id, best_ssim_id, worst_ssim_id = [], [], [], []
        best_psnr_id, worst_psnr_id, best_ssim_id, worst_ssim_id = [], [], [], []
        for bid in range(configs.batch_size):
            seq_id = batch_id * configs.batch_size + bid
            avg_psnr_seq = []
            avg_ssim_seq = []
            avg_lpips_seq = []
            for sid in range(configs.num_samples):
                sid_in_batch = sid*configs.batch_size + bid
                mse_sample = np.square(gt[bid] - gx[sid_in_batch]).sum()
                #print(mse_sample)
                # ssim_sample, psnr_sample, mae_sample, sharp_sample = 0, 0, 0, 0
                ssim_sample, psnr_sample, mae_sample, sharp_sample, lpips_sample = 0, 0, 0, 0, 0
                for i in range(output_length):
                    gt_frm = gt[bid, i]
                    gx_frm = gx[sid*configs.batch_size + bid, i]
                    #mae_sample += metrics.batch_mae_frame_float(gx_frm, gt_frm)
                    img_mse[i] += np.square(gt[bid,i] - gx[sid_in_batch,i]).sum()
                    gt_frm = np.uint8(gt_frm * 255)
                    gx_frm = np.uint8(gx_frm * 255)
                    #psnr_sample += metrics.batch_psnr(gx_frm, gt_frm)
                    #sharp_sample += np.max(cv2.convertScaleAbs(cv2.Laplacian(gx_frm[0], 3)))
                    score, _ = compare_ssim(gx_frm, gt_frm, full=True, multichannel=True)
                    # ssim_sample += score
                    ssim_sample += score/output_length
                    ssim[i]+=score

                    psnr_tmp = compare_psnr(gx_frm, gt_frm, data_range=255)
                    psnr_sample += psnr_tmp/output_length
                    img_psnr[i] += psnr_tmp

                    # for lpips
                    gt_frm_lpips = img_gt_lpips[bid, i].unsqueeze(0)
                    gx_frm_lpips = img_gen_lpips[sid*batch_size + bid, i].unsqueeze(0)
                    lpips_tmp = loss_fn.forward(gt_frm_lpips, gx_frm_lpips)
                    lpips_sample += lpips_tmp[0].detach().cpu().numpy() / output_length

                avg_psnr_seq.append(psnr_sample)
                avg_ssim_seq.append(ssim_sample)
                avg_lpips_seq.append(lpips_sample)

                '''[b1s1 b2s1 b3s1, b1s2 b2s2 b3s2]'''
                if sid == 0:
                    
                    best_ssim.append(ssim_sample)
                    avg_ssim.append(ssim_sample)
                    worst_ssim.append(ssim_sample)
                    best_ssim_id.append(sid_in_batch)
                    worst_ssim_id.append(sid_in_batch)

                    
                    best_psnr.append(psnr_sample)
                    avg_psnr.append(psnr_sample)
                    worst_psnr.append(psnr_sample)
                    best_psnr_id.append(sid_in_batch)
                    worst_psnr_id.append(sid_in_batch)
                    best_last_psnr.append(img_psnr[output_length-1])
                    worst_last_psnr.append(img_psnr[output_length-1])

                else:
                    if best_psnr[-1] > psnr_sample:
                        best_psnr[-1] = psnr_sample
                        best_psnr_id[-1] = sid_in_batch
                    if worst_psnr[-1] < psnr_sample:
                        worst_psnr[-1] = psnr_sample
                        worst_psnr_id[-1] = sid_in_batch
                    if worst_last_psnr[-1]<img_psnr[output_length-1]:
                        worst_last_psnr[-1]=img_psnr[output_length-1]
                    avg_psnr[seq_id] += psnr_sample

                    if best_ssim[-1] < ssim_sample:
                        best_ssim[-1] = ssim_sample
                        best_ssim_id[-1] = sid_in_batch
                    if worst_ssim[-1] > ssim_sample:
                        worst_ssim[-1] = ssim_sample
                        worst_ssim_id[-1] = sid_in_batch
                    if best_last_psnr[-1]>img_psnr[output_length-1]:
                        best_last_psnr[-1]=img_psnr[output_length-1]
                    avg_ssim[seq_id] += ssim_sample

                    # avg_lpips[seq_id] += lpips_sample

            # calculate the mean and std for each seq
            psnr_mean_seq = np.mean(np.asarray(avg_psnr_seq, dtype=np.float32))
            psnr_std_seq = np.std(np.asarray(avg_psnr_seq, dtype=np.float32))
            ssim_mean_seq = np.mean(np.asarray(avg_ssim_seq, dtype=np.float32))
            ssim_std_seq = np.std(np.asarray(avg_ssim_seq, dtype=np.float32))
            lpips_mean_seq = np.mean(np.asarray(avg_lpips_seq, dtype=np.float32))
            lpips_std_seq = np.std(np.asarray(avg_lpips_seq, dtype=np.float32))

            avg_psnr_all.append(np.asarray([psnr_mean_seq, psnr_std_seq]))
            avg_ssim_all.append(np.asarray([ssim_mean_seq, ssim_std_seq]))
            avg_lpips_all.append(np.asarray([lpips_mean_seq, lpips_std_seq]))

            mean_psnr_all.append(psnr_mean_seq)

            #save prediction examples

            

        batch_id = batch_id + 1  # batch_size=1 for test
        test_input_handle.next()

    # Average BEST and WORST predictions over all seqs
    # print(len(avg_mse))
    # print(batch_id * configs.batch_size*configs.num_samples)
    best_psnr = np.mean(np.asarray(best_psnr, dtype=np.float32)) #/ output_length
    avg_psnr = np.mean(np.asarray(avg_psnr, dtype=np.float32))/configs.num_samples
    worst_psnr = np.mean(np.asarray(worst_psnr, dtype=np.float32)) #/ output_length

    best_last_psnr=np.mean(np.asarray(best_last_psnr, dtype=np.float32))
    worst_last_psnr=np.mean(np.asarray(worst_last_psnr, dtype=np.float32))
    print('best psnr last frame: ' + str(best_last_psnr))
    print('worst psnr last frame: ' + str(worst_last_psnr))

    print('best psnr per frame: ' + str(best_psnr))
    print('avg psnr per seq: ' + str(avg_psnr))
    for i in range(configs.total_length - configs.input_length):
        print(img_psnr[i] / (batch_id * batch_size*configs.num_samples))
    print('worst psnr per frame: ' + str(worst_psnr))

    best_ssim = np.mean(np.asarray(best_ssim, dtype=np.float32)) #/ output_length
    avg_ssim = np.mean(np.asarray(avg_ssim, dtype=np.float32))/ (configs.num_samples)
    worst_ssim = np.mean(np.asarray(worst_ssim, dtype=np.float32)) #/ output_length

    print('best ssim per frame: ' + str(best_ssim))
    print('avg ssim per frame: ' + str(avg_ssim))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i]/ (batch_id * batch_size*configs.num_samples))
    print('worst ssim per frame: ' + str(worst_ssim))


    avg_psnr_all_mean = np.mean(np.asarray(avg_psnr_all), 0)
    avg_ssim_all_mean = np.mean(np.asarray(avg_ssim_all), 0)
    avg_lpips_all_mean = np.mean(np.asarray(avg_lpips_all), 0)

    print('Final avg_psnr_all:' + str(avg_psnr_all_mean))
    print('Final avg_ssim_all:' + str(avg_ssim_all_mean))
    print('Final avg_lpips_all:' + str(avg_lpips_all_mean))


    if configs.model_name=='svg_curl' or configs.model_name=='svg_curl_replay':
        cluster_accuracy=allright_cat_num/(batch_size * batch_id*configs.num_samples)
        print('cluster_accuracy'+str(cluster_accuracy))
    ssim = np.asarray(ssim, dtype=np.float32) / (batch_size * batch_id*configs.num_samples)
    record = dict()
    record["avg_psnr_per_seq"] = avg_psnr
    record["img_psnr_per_frame"] = []
    for i in range(configs.total_length - configs.input_length):
        record["img_psnr_per_frame"].append(img_psnr[i] / (batch_id * batch_size*configs.num_samples))
    record["avg_ssim_per_seq"] = avg_ssim
    record["img_ssim_per_frame"] = []
    for i in range(configs.total_length - configs.input_length):
        record["img_ssim_per_frame"].append(ssim[i] / (batch_id * batch_size*configs.num_samples))

    # avg_lpips = np.mean(np.asarray(avg_lpips, dtype=np.float32))/ (configs.num_samples)
    # record["avg_lpips_per_seq"] = avg_lpips
    # record["img_lpips_per_frame"] = []
    # for i in range(configs.total_length - configs.input_length):
    #     record["img_lpips_per_frame"].append(img_lpips[i] / (batch_id * batch_size*configs.num_samples))

    record['Final avg_psnr_all']=avg_psnr_all_mean
    record['Final avg_ssim_all']=avg_ssim_all_mean
    record['Final mean_psnr_all']=np.asarray(mean_psnr_all)
    record['Final avg_lpips_all']=avg_lpips_all_mean
    return record

def test_kth_relabel(model, test_model,test_input_handle, configs, itr,category=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs.gen_frm_dir, 'itr_'+str(itr))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    test_input_handle.begin(do_shuffle=False)
    test_input_handle.minibatch_size=8

    batch_id = 0
    best_mse, best_ssim, best_psnr, best_mae, best_sharp = [], [], [], [], []
    avg_mse, avg_ssim, avg_psnr, avg_mae, avg_sharp = [], [], [], [], []
    worst_mse, worst_ssim, worst_psnr, worst_mae, worst_sharp = [], [], [], [], []

    best_last_psnr,worst_last_psnr=[],[]
    img_mse, ssim, img_psnr, img_lpips = [], [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        img_psnr.append(0)
        img_lpips.append(0)

    allright_cat_num=0

    batch_size = 8 #configs.batch_size
    avg_psnr_all = []
    avg_ssim_all = []
    avg_lpips_all = []

    mean_psnr_all = []

    new_label = []
    total_num = test_input_handle.total()
    while (test_input_handle.no_batch_left() == False ):

        sample=np.zeros(
        (batch_size* 1,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel)) 

        test_ims ,test_cat, _= test_input_handle.get_batch()

        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        new_cat = model.test_relabel(test_model,test_dat, sample,category)
        new_label.append(np.array(new_cat))

        batch_id = batch_id + 1
        test_input_handle.next()

        if np.mod(batch_id, 100) == 0:
            print('relabel about %f' % (batch_id * batch_size / total_num))
    
    new_label = np.concatenate(new_label, 0)
    new_label = np.expand_dims(new_label, 1)

    return new_label