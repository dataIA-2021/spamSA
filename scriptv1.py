{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9e14a4a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "\n",
    "df = pd.read_csv(\"spam.csv\", encoding='latin-1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "04a78853",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "      <th>Unnamed: 3</th>\n",
       "      <th>Unnamed: 4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>ham</td>\n",
       "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>ham</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>ham</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>ham</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        v1                                                 v2 Unnamed: 2  \\\n",
       "0      ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
       "1      ham                      Ok lar... Joking wif u oni...        NaN   \n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
       "3      ham  U dun say so early hor... U c already then say...        NaN   \n",
       "4      ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
       "...    ...                                                ...        ...   \n",
       "5567  spam  This is the 2nd time we have tried 2 contact u...        NaN   \n",
       "5568   ham              Will Ì_ b going to esplanade fr home?        NaN   \n",
       "5569   ham  Pity, * was in mood for that. So...any other s...        NaN   \n",
       "5570   ham  The guy did some bitching but I acted like i'd...        NaN   \n",
       "5571   ham                         Rofl. Its true to its name        NaN   \n",
       "\n",
       "     Unnamed: 3 Unnamed: 4  \n",
       "0           NaN        NaN  \n",
       "1           NaN        NaN  \n",
       "2           NaN        NaN  \n",
       "3           NaN        NaN  \n",
       "4           NaN        NaN  \n",
       "...         ...        ...  \n",
       "5567        NaN        NaN  \n",
       "5568        NaN        NaN  \n",
       "5569        NaN        NaN  \n",
       "5570        NaN        NaN  \n",
       "5571        NaN        NaN  \n",
       "\n",
       "[5572 rows x 5 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "54f397f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.rename({\"Unnamed: 2\":\"a\",\"Unnamed: 3\":\"b\",\"Unnamed: 4\":\"c\"}, axis=\"columns\", inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10443447",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5b914c09",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['a'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "428e4614",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['b'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "eb2183c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['c'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fd5583d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>ham</td>\n",
       "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>ham</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>ham</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>ham</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        v1                                                 v2\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
       "...    ...                                                ...\n",
       "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
       "5568   ham              Will Ì_ b going to esplanade fr home?\n",
       "5569   ham  Pity, * was in mood for that. So...any other s...\n",
       "5570   ham  The guy did some bitching but I acted like i'd...\n",
       "5571   ham                         Rofl. Its true to its name\n",
       "\n",
       "[5572 rows x 2 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f2c42d2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>ham</td>\n",
       "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>ham</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>ham</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>ham</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        v1                                                 v2\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
       "...    ...                                                ...\n",
       "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
       "5568   ham              Will Ì_ b going to esplanade fr home?\n",
       "5569   ham  Pity, * was in mood for that. So...any other s...\n",
       "5570   ham  The guy did some bitching but I acted like i'd...\n",
       "5571   ham                         Rofl. Its true to its name\n",
       "\n",
       "[5572 rows x 2 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "fa436e96",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns = ['label','message']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5f1eb602",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>ham</td>\n",
       "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>ham</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>ham</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>ham</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     label                                            message\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
       "...    ...                                                ...\n",
       "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
       "5568   ham              Will Ì_ b going to esplanade fr home?\n",
       "5569   ham  Pity, * was in mood for that. So...any other s...\n",
       "5570   ham  The guy did some bitching but I acted like i'd...\n",
       "5571   ham                         Rofl. Its true to its name\n",
       "\n",
       "[5572 rows x 2 columns]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "95b9dcb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def countspam(code):\n",
    "    n=0\n",
    "    countsp = 0\n",
    "    countha = 0\n",
    "    for line in df['message']: # Parcours des lignes\n",
    "        x = re.findall(code, line)\n",
    "        if x:\n",
    "            print(n, 'findall:', x)#commenter les prints pour de la visibilité\n",
    "            print(df['label'][n])\n",
    "            print(df['message'][n])\n",
    "        \n",
    "            if df['label'][n]== 'spam':\n",
    "                countsp +=1\n",
    "            else:\n",
    "                countha +=1     \n",
    "        n+=1\n",
    "    print('Nombre de Spam :', countsp)\n",
    "    print('Nombre de Ham :', countha)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a7f9d37e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Nombre de Spam : 0\n",
      "Nombre de Ham : 0\n"
     ]
    }
   ],
   "source": [
    "countspam('Bulk')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0a8fb0fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "grouped = df.groupby(df.label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "453e35c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "spam = grouped.get_group(\"spam\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8539bf87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>spam</td>\n",
       "      <td>FreeMsg Hey there darling it's been 3 week's n...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>spam</td>\n",
       "      <td>WINNER!! As a valued network customer you have...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>spam</td>\n",
       "      <td>Had your mobile 11 months or more? U R entitle...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>spam</td>\n",
       "      <td>SIX chances to win CASH! From 100 to 20,000 po...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5537</th>\n",
       "      <td>spam</td>\n",
       "      <td>Want explicit SEX in 30 secs? Ring 02073162414...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5540</th>\n",
       "      <td>spam</td>\n",
       "      <td>ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5547</th>\n",
       "      <td>spam</td>\n",
       "      <td>Had your contract mobile 11 Mnths? Latest Moto...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5566</th>\n",
       "      <td>spam</td>\n",
       "      <td>REMINDER FROM O2: To get 2.50 pounds free call...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>747 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     label                                            message\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "5     spam  FreeMsg Hey there darling it's been 3 week's n...\n",
       "8     spam  WINNER!! As a valued network customer you have...\n",
       "9     spam  Had your mobile 11 months or more? U R entitle...\n",
       "11    spam  SIX chances to win CASH! From 100 to 20,000 po...\n",
       "...    ...                                                ...\n",
       "5537  spam  Want explicit SEX in 30 secs? Ring 02073162414...\n",
       "5540  spam  ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...\n",
       "5547  spam  Had your contract mobile 11 Mnths? Latest Moto...\n",
       "5566  spam  REMINDER FROM O2: To get 2.50 pounds free call...\n",
       "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
       "\n",
       "[747 rows x 2 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2028e21b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7e36c989",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      label  message\n",
       "0     False    False\n",
       "1     False    False\n",
       "2     False    False\n",
       "3     False    False\n",
       "4     False    False\n",
       "...     ...      ...\n",
       "5567  False    False\n",
       "5568  False    False\n",
       "5569  False    False\n",
       "5570  False    False\n",
       "5571  False    False\n",
       "\n",
       "[5572 rows x 2 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isna(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d1f183e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "63fc1965",
   "metadata": {},
   "outputs": [],
   "source": [
    "label = df['label']\n",
    "label_encod = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "1ce17d86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LabelEncoder()"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "label_encod.fit(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d7a34d1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = label_encod.transform(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae8a3606",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5c78ea8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f69b9830",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
