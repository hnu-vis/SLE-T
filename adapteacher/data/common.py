
import logging
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset


class MapDatasetTwoCrop(MapDataset):














    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class AspectRatioGroupedDatasetTwoCrop(AspectRatioGroupedDataset):












    def __init__(self, dataset, batch_size):






        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._buckets_key = [[] for _ in range(2)]
        
        

    def __iter__(self):
        for d in self.dataset:
            
            
            w, h = d[0]["width"], d[0]["height"]
            bucket_id = 0 if w > h else 1

            
            bucket = self._buckets[bucket_id]
            bucket.append(d[0])

            
            buckets_key = self._buckets_key[bucket_id]
            buckets_key.append(d[1])
            if len(bucket) == self.batch_size:
                yield (bucket[:], buckets_key[:])
                del bucket[:]
                del buckets_key[:]


class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):












    def __init__(self, dataset, batch_size):








        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = [[] for _ in range(2)]
        self._label_buckets_key = [[] for _ in range(2)]
        self._unlabel_buckets = [[] for _ in range(2)]
        self._unlabel_buckets_key = [[] for _ in range(2)]
        
        

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):
            
            
            

            
            
            
            if len(label_bucket) != self.batch_size_label:
                w, h = d_label[0]["width"], d_label[0]["height"]
                label_bucket_id = 0 if w > h else 1
                label_bucket = self._label_buckets[label_bucket_id]
                label_bucket.append(d_label[0])
                label_buckets_key = self._label_buckets_key[label_bucket_id]
                label_buckets_key.append(d_label[1])

            if len(unlabel_bucket) != self.batch_size_unlabel:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
                unlabel_bucket.append(d_unlabel[0])
                unlabel_buckets_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_buckets_key.append(d_unlabel[1])

            
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                
                yield (
                    label_bucket[:],
                    label_buckets_key[:],
                    unlabel_bucket[:],
                    unlabel_buckets_key[:],
                )
                del label_bucket[:]
                del label_buckets_key[:]
                del unlabel_bucket[:]
                del unlabel_buckets_key[:]
