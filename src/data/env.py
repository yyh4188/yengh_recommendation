import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from .utils import (
    prepare_batch_static_size,
    make_items_tensor
)


# 恢复原库的数据处理管道类
class DataFuncKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def keys(self):
        return self.kwargs.keys()

    def get(self, name: str):
        if name not in self.kwargs:
            example = """
                # example on how to use kwargs:
                def prepare_dataset(args, args_mut):
                    args.set_kwarg('{}', your_value) # set kwargs for your functions here!
                    pipeline = [recnn.data.truncate_dataset, recnn.data.prepare_dataset]
                    recnn.data.build_data_pipeline(pipeline, args, args_mut)
            """
            raise AttributeError(
                "No kwarg with name {} found!\n{}".format(name, example.format(example))
            )
        return self.kwargs[name]

    def set(self, name: str, value):
        self.kwargs[name] = value


class DataFuncArgsMut:
    def __init__(
        self, df, base, users, user_dict
    ):
        self.base = base
        self.users = users
        self.user_dict = user_dict
        self.df = df


def try_progress_apply(dataframe, function):
    try:
        return dataframe.progress_apply(function)
    except AttributeError:
        return dataframe.apply(function)


def prepare_dataset(args_mut: DataFuncArgsMut, kwargs: DataFuncKwargs):
    """
    Basic prepare dataset function. Automatically makes index linear, in ml20 movie indices look like:
    [1, 34, 123, 2000], recnn makes it look like [0,1,2,3] for you.
    """

    # get args
    frame_size = kwargs.get("frame_size")
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df

    # rating range mapped from [0, 5] to [-5, 5]
    df["rating"] = try_progress_apply(df["rating"], lambda i: 2 * (i - 2.5))
    # id's tend to be inconsistent and sparse so they are remapped here
    df["movieId"] = try_progress_apply(df["movieId"], lambda x: key_to_id.get(x, 0))
    users = df[["userId", "movieId"]].groupby(["userId"]).size()
    users = users[users > frame_size].index

    user_dict = {}
    for user_id, group in df.groupby('userId'):
        if user_id in users:
            group = group.sort_values('timestamp')
            items = group['movieId'].values
            ratings_vals = group['rating'].values
            user_dict[user_id] = {
                'items': items,
                'ratings': ratings_vals
            }

    args_mut.users = users
    args_mut.user_dict = user_dict
    args_mut.df = df


class UserDataset(Dataset):

    """
    Low Level API: dataset class user: [items, ratings], Instance of torch.DataSet
    """

    def __init__(self, users, user_dict):
        """

        :param users: integer list of user_id. Useful for train/test splitting
        :type users: list<int>.
        :param user_dict: dictionary of users with user_id as key and [items, ratings] as value
        :type user_dict: (dict{ user_id<int>: dict{'items': list<int>, 'ratings': list<int>} }).

        """

        self.users = users
        self.user_dict = user_dict

    def __len__(self):
        """
        useful for tqdm, consists of a single line:
        return len(self.users)
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        getitem is a function where non linear user_id maps to a linear index. For instance in the ml20m dataset,
        there are big gaps between neighbouring user_id. getitem removes these gaps, optimizing the speed.

        :param idx: index drawn from range(0, len(self.users)). User id can be not linear, idx is.
        :type idx: int

        :returns:  dict{'items': list<int>, rates:list<int>, sizes: int}
        """
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group["items"][:]
        rates = group["ratings"][:]
        size = items.shape[0]
        return {"items": items, "rates": rates, "sizes": size, "users": idx}


class EnvBase:

    """
    Misc class used for serializing
    """

    def __init__(self):
        self.train_user_dataset = None
        self.test_user_dataset = None
        self.embeddings = None
        self.key_to_id = None
        self.id_to_key = None


class DataPath:

    """
    [New!] Path to your data. Note: cache is optional. It saves EnvBase as a pickle
    """

    def __init__(
        self,
        base: str,
        ratings: str,
        embeddings: str,
        cache: str = "",
        use_cache: bool = True,
    ):
        self.ratings = base + ratings
        self.embeddings = base + embeddings
        self.cache = base + cache
        self.use_cache = use_cache


class Env:

    """
    Env abstract class
    """

    def __init__(
        self,
        path: DataPath,
        prepare_dataset=prepare_dataset,
        embed_batch=None,
        **kwargs
    ):

        """
        .. note::
            embeddings need to be provided in {movie_id: torch.tensor} format!

        :param path: DataPath to where item embeddings are stored.
        :type path: DataPath
        :param test_size: ratio of users to use in testing. Rest will be used for training/validation
        :type test_size: int
        :param min_seq_size: (use as kwarg) filter users: len(user.items) > min seq size
        :type min_seq_size: int
        :param embed_batch: function to apply embeddings to batch. Can be set to yield continuous/discrete state/action
        :type embed_batch: function
        """

        from .utils import batch_tensor_embeddings

        self.base = EnvBase()
        self.embed_batch = embed_batch if embed_batch is not None else batch_tensor_embeddings
        self.prepare_dataset = prepare_dataset
        if path.use_cache and os.path.isfile(path.cache):
            self.load_env(path.cache)
        else:
            self.process_env(path, **kwargs)
            if path.use_cache:
                self.save_env(path.cache)

    def process_env(self, path: DataPath, **kwargs):
        if "frame_size" in kwargs.keys():
            frame_size = kwargs["frame_size"]
        else:
            frame_size = 10

        if "test_size" in kwargs.keys():
            test_size = kwargs["test_size"]
        else:
            test_size = 0.05

        movie_embeddings_key_dict = pickle.load(open(path.embeddings, "rb"))
        (
            self.base.embeddings,
            self.base.key_to_id,
            self.base.id_to_key,
        ) = make_items_tensor(movie_embeddings_key_dict)
        ratings = pd.read_csv(path.ratings)

        process_kwargs = DataFuncKwargs(
            frame_size=frame_size,  # remove when lstm gets implemented
        )

        process_args_mut = DataFuncArgsMut(
            df=ratings,
            base=self.base,
            users=None,  # will be set later
            user_dict=None,  # will be set later
        )

        self.prepare_dataset(process_args_mut, process_kwargs)
        self.base = process_args_mut.base
        users = process_args_mut.users
        user_dict = process_args_mut.user_dict

        train_users, test_users = train_test_split(users, test_size=test_size)
        train_users = sort_users_itemwise(user_dict, train_users)[2:]
        test_users = sort_users_itemwise(user_dict, test_users)
        self.base.train_user_dataset = UserDataset(train_users, user_dict)
        self.base.test_user_dataset = UserDataset(test_users, user_dict)



    def load_env(self, where: str):
        self.base = pickle.load(open(where, "rb"))

    def save_env(self, where: str):
        # 创建目录结构
        import os
        os.makedirs(os.path.dirname(where), exist_ok=True)
        pickle.dump(self.base, open(where, "wb"))


class FrameEnv(Env):
    """
    Static length user environment.
    """

    def __init__(
        self, path, frame_size=10, batch_size=25, num_workers=1, *args, **kwargs
    ):

        """
        :param embeddings: path to where item embeddings are stored.
        :type embeddings: str
        :param ratings: path to the dataset that is similar to the ml20m
        :type ratings: str
        :param frame_size: len of a static sequence, frame
        :type frame_size: int

        p.s. you can also provide **pandas_conf in the arguments.

        It is useful if you dataset columns are different from ml20::

            pandas_conf = {user_id='userId', rating='rating', item='movieId', timestamp='timestamp'}
            env = FrameEnv(embed_dir, rating_dir, **pandas_conf)

        """

        kwargs["frame_size"] = frame_size
        super(FrameEnv, self).__init__(
            path, *args, **kwargs
        )

        self.frame_size = frame_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataloader = DataLoader(
            self.base.train_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper,
        )

        self.test_dataloader = DataLoader(
            self.base.test_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper,
        )

    def prepare_batch_wrapper(self, x):
        batch = prepare_batch_static_size(
            x,
            self.base.embeddings,
            embed_batch=self.embed_batch,
            frame_size=self.frame_size,
        )
        return batch

    def train_batch(self):
        """ Get batch for training """
        return next(iter(self.train_dataloader))

    def test_batch(self):
        """ Get batch for testing """
        return next(iter(self.test_dataloader))


def sort_users_itemwise(user_dict, users):
    return (
        pd.Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users]))
        .sort_values(ascending=False)
        .index
    )
